"""A faster TRICERATOPS: the published code with its light-curve step replaced, and the same evidences.

TRICERATOPS (Giacalone et al. 2021, MIT) spends 83-94% of a run in three functions that score each draw's light curve.
install() swaps those three (lnL_TP_p, lnL_EB_p, lnL_EB_twin_p) for the ones here; the priors, scenarios and evidences
stay the published code, so with the same random seed the same draws are scored. What's different:

  - A draw's secondary eclipse is tested first. Stock simulates the whole primary, then throws away 97% of EB draws on
    that test.
  - Bins outside the transit cost nothing: chi-square is the flat line's, plus a correction from the bins in transit.
  - Bins go from the transit's centre outwards, and a draw stops as soon as it provably can't come within 100 of the best
    lnL. That assumes no prior below e^-100 on the best draw: true without contrast curves, which pipeline/fpp.py doesn't use.
  - The flux needs no elliptic integrals. A small occulter inside the limb has a closed form (the mean of the limb
    darkening over a disc, by Pizzetti's series). Otherwise the limb-darkened star is integrated directly: the inner disc
    fully covered in closed form, and over the annuli partly covered, the covered share of each annulus (an arccos) by
    Gauss-Legendre in an angle that removes the square-root ends. Checked to 6 digits against brute force and PyTransit,
    at k up to 1.5. Not derived from PyTransit (GPL).
  - The orbit is Kepler's equation, solved. PyTransit expands it in a Taylor series about the transit, so the two agree
    at e = 0.3 (1e-8 in flux) and part at e = 0.9, where stock's light curves are off. Those draws fit so badly that it
    doesn't change an evidence, but it's why lnL can differ there.
  - The per-draw limb-darkening loop in the scenarios with a companion becomes one lookup per distinct star.
  - One upstream quirk is kept on purpose, for the same numbers: the parallel EB model multiplies every k below 1 by
    0.999 (`k[(k - 1.0) < 1e-6]`, no abs(); the serial model has the abs()).

Run: python -m pipeline.tri_fast     parity and speed on the golden calls (tools/fpp_golden.py)
"""
import inspect
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit, prange

NODES = 12  # Gauss-Legendre nodes across the partly covered annuli
_x, _w = np.polynomial.legendre.leggauss(NODES)
_phi = (_x + 1) * np.pi / 2
S = (1 - np.cos(_phi)) / 2  # 0..1 across the band, bunched at both ends
DS = np.sin(_phi) / 2 * _w * np.pi / 2
SECONDARY_T = np.linspace(-0.05, 0.05, 25)
PILOT, MARGIN = 2000, 100.0  # a draw this far above the pilot's best lnL carries e^-100 of its weight
SMALL_K, CLEAR_OF_LIMB = 0.1, 0.9  # the closed form holds for k below this, with z + k below that
Rsun, Rearth = 6.957e10, 6.3781e8  # replaced by TRICERATOPS's own constants in install()
CACHE = None  # the folder for each star's setup, when install() is given one
POOL = ThreadPoolExecutor(2)
PENDING = {}  # the Gaia queries running behind the rest of the setup, by the file each will write


@njit(cache=True)
def blocked(z, k, u1, u2, s, ds):
    """The share of a quadratically limb-darkened star's light an occulter of radius k covers at distance z (both in
    stellar radii). I(mu) = c0 + c1 mu + c2 mu^2."""
    if z >= 1 + k:
        return 0.0
    c0, c1, c2 = 1 - u1 - u2, u1 + 2 * u2, -u2
    total = 1 - u1 / 3 - u2 / 6
    k2, z2 = k * k, z * z
    if k < SMALL_K and z + k < CLEAR_OF_LIMB:  # the mean of I over the occulter
        m2 = 1 - z2
        m = np.sqrt(m2)
        lap = -(2 - z2) / (m * m2)  # the Laplacian of mu, then of that
        lap2 = (z2 * z2 - 8 * z2 - 8) / (m * m2 * m2 * m2)
        mean_mu = m + k2 / 8 * lap + k2 * k2 / 192 * lap2
        return k2 * (c0 + c1 * mean_mu + c2 * (m2 - k2 / 2)) / total
    z = max(z, 1e-9)
    inner = min(max(k - z, 0.0), 1.0)
    inner2 = inner * inner
    mu_in = np.sqrt(max(1 - inner2, 0.0))
    out = c0 * inner2 + c1 * 2 / 3 * (1 - mu_in * mu_in * mu_in) + c2 * (inner2 - inner2 * inner2 / 2)
    lo, hi = abs(z - k), min(z + k, 1.0)
    width = hi - lo
    if width > 0:
        band = 0.0
        for i in range(len(s)):
            r = max(lo + width * s[i], 1e-12)
            r2 = r * r
            mu = np.sqrt(max(1 - r2, 0.0))
            c = (r2 + z2 - k2) / (2 * r * z)
            c = min(max(c, -1.0), 1.0)
            band += (c0 + c1 * mu + c2 * mu * mu) * 2 * r * np.arccos(c) * ds[i]
        out += width * band / np.pi
    return out / total


@njit(cache=True)
def conjunction(ecc, w):
    """The mean anomaly at conjunction."""
    f = np.pi / 2 - w
    e = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(f / 2), np.sqrt(1 + ecc) * np.cos(f / 2))
    return e - ecc * np.sin(e)


@njit(cache=True)
def sky_z(t, n, m0, a, sin_i, ecc, w, q, ea):
    """Star-companion distance in the sky plane at t, negative behind the star, and the eccentric anomaly (a warm start
    for the next t). q = sqrt((1 + e) / (1 - e))."""
    mean = n * t + m0
    for _ in range(12):
        step = (ea - ecc * np.sin(ea) - mean) / (1 - ecc * np.cos(ea))
        ea -= step
        if abs(step) < 1e-12:
            break
    half = np.tan(ea / 2)
    nu = 2 * np.arctan(q * half)
    r = a * (1 - ecc * np.cos(ea))
    s = np.sin(w + nu)
    z = r * np.sqrt(max(1 - (sin_i * s) ** 2, 0.0))
    return (z if s > 0 else -z), ea


@njit(cache=True, parallel=True)
def secondary_ok(k, p, a, inc, ecc, w, u1, u2, third, fe_share, sigma, t, s, ds):
    """Whether each draw's secondary eclipse is shallow enough not to have been seen (depth below 1.5 sigma). third is
    the light of everything but the eclipsed body, over its own, and fe_share rescales to the system's light."""
    n = len(k)
    ok = np.zeros(n, dtype=np.bool_)
    for d in prange(n):
        e = ecc[d]
        m0 = conjunction(e, w[d])
        q = np.sqrt((1 + e) / (1 - e))
        sin_i = np.sin(inc[d])
        rate = 2 * np.pi / p[d]
        lowest = 1.0
        ea = rate * t[0] + m0
        for j in range(len(t)):
            z, ea = sky_z(t[j], rate, m0, a[d], sin_i, e, w[d], q, ea)
            if z > 0:
                lowest = min(lowest, 1 - blocked(z, k[d], u1[d], u2[d], s, ds))
        ok[d] = fe_share[d] * (1 - lowest) / (1 + third[d]) < 1.5 * sigma
    return ok


@njit(cache=True)
def orbit_knots(t, rate, m0, a, cos_i, sin_i, ecc, w, ea):
    """Sky-plane position and velocity at t, by Kepler's equation; depth along the line of sight, positive in front."""
    mean = rate * t + m0
    for _ in range(12):
        step = (ea - ecc * np.sin(ea) - mean) / (1 - ecc * np.cos(ea))
        ea -= step
        if abs(step) < 1e-12:
            break
    sin_e, cos_e = np.sin(ea), np.cos(ea)
    root = np.sqrt(1 - ecc * ecc)
    away = 1 - ecc * cos_e
    theta = w + np.arctan2(root * sin_e, cos_e - ecc)
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    r = a * away
    r_dot = a * ecc * sin_e * rate / away
    nu_dot = rate * root / (away * away)
    return (r * cos_t, r * sin_t * cos_i, r_dot * cos_t - r * sin_t * nu_dot,
            (r_dot * sin_t + r * cos_t * nu_dot) * cos_i, r * sin_t * sin_i, ea)


KNOTS, KNOTS_HIGH_E, HIGH_E = 13, 49, 0.5
ROOT_06 = 0.6 ** 0.5  # three-point Gauss-Legendre, for bins with no contact in them
BLOCK = 512  # draws per task: a task's scratch arrays are allocated once


@njit(cache=True)
def position(t, window, h, knots, xs, ys, vx, vy):
    """The interpolated orbit at t: squared distance, and x (its sign says which side of the star)."""
    v = min(max((t + window) / h, 0.0), knots - 1 - 1e-9)
    j = int(v)
    f = v - j
    g = 1 - f
    c0, c1, c2, c3 = (1 + 2 * f) * g * g, f * g * g * h, f * f * (3 - 2 * f), -f * f * g * h
    x = c0 * xs[j] + c1 * vx[j] + c2 * xs[j + 1] + c3 * vx[j + 1]
    y = c0 * ys[j] + c1 * vy[j] + c2 * ys[j + 1] + c3 * vy[j + 1]
    return x * x + y * y, x


@njit(cache=True)
def sample_flux(t, window, h, knots, xs, ys, vx, vy, front, reach, k, u1, u2, s, ds):
    """The star's light at t: nothing dims it from behind, and a long window can reach there at high e."""
    if abs(t) >= window:
        return 1.0
    j = min(int((t + window) / h), knots - 2)
    if front[j] <= 0 and front[j + 1] <= 0:
        return 1.0
    z2, _ = position(t, window, h, knots, xs, ys, vx, vy)
    if z2 >= reach:
        return 1.0
    return 1 - blocked(np.sqrt(z2), k, u1, u2, s, ds)


@njit(cache=True)
def region(z2, inner, reach):
    return 0 if z2 < inner else 1 if z2 < reach else 2


@njit(cache=True)
def transit_change(time, data, flat, scale, limit, room, distance, half_exp, edges, weights, k, p, a, inc, e, w, u1, u2, third,
                   xs, ys, vx, vy, front, s, ds):
    """One draw: what its transit adds to the flat line's chi-square, or inf once it provably can't get under limit."""
    ns = len(edges)
    m0 = conjunction(e, w)
    sin_i, cos_i = np.sin(inc), np.cos(inc)
    rate = 2 * np.pi / p
    speed = rate * a * (1 + e * np.sin(w)) / np.sqrt(1 - e * e)  # across the sky at conjunction
    limit_t = p / 4
    window = min(1.6 * (1 + k) / speed, limit_t)
    reach = (1 + k) ** 2
    inner = (1 - k) ** 2 if k < 1 else -1.0
    while window < limit_t:  # high e: the speed changes during the transit
        xa, ya, _, _, _, _ = orbit_knots(window, rate, m0, a, cos_i, sin_i, e, w, rate * window + m0)
        xb, yb, _, _, _, _ = orbit_knots(-window, rate, m0, a, cos_i, sin_i, e, w, -rate * window + m0)
        if xa * xa + ya * ya > reach and xb * xb + yb * yb > reach:
            break
        window = min(window * 1.5, limit_t)
    knots = KNOTS if e < HIGH_E else KNOTS_HIGH_E
    h = 2 * window / (knots - 1)
    ea = -rate * window + m0
    for j in range(knots):
        xs[j], ys[j], vx[j], vy[j], front[j], ea = orbit_knots(-window + j * h, rate, m0, a, cos_i, sin_i, e, w, ea)
    change = 0.0
    last = int(np.searchsorted(distance, window + half_exp, side="right"))
    for b in range(last):
        if (flat + change - (room[last] - room[b])) * scale > limit:
            return np.inf
        t = time[b]
        z_a, x_a = position(t - half_exp, window, h, knots, xs, ys, vx, vy)
        z_b, x_b = position(t + half_exp, window, h, knots, xs, ys, vx, vy)
        same_side = x_a * x_b > 0
        ra, rb = region(z_a, inner, reach), region(z_b, inner, reach)
        if ra == 2 and rb == 2 and same_side:
            continue  # the whole exposure is off the disc
        f = 0.0
        if ra == rb and same_side:
            for i in range(3):
                f += weights[i] * sample_flux(t + (i - 1) * ROOT_06 * half_exp, window, h, knots, xs, ys, vx, vy, front, reach,
                                              k, u1, u2, s, ds)
        else:
            for i in range(ns):
                f += sample_flux(t + edges[i], window, h, knots, xs, ys, vx, vy, front, reach, k, u1, u2, s, ds) / ns
        if f < 1:
            model = (f + third) / (1 + third)
            change += (data[b] - model) ** 2 - (data[b] - 1) ** 2
    return change


@njit(cache=True, parallel=True)
def half_chi2(time, data, sigma, exptime, ns, k, p, a, inc, ecc, w, u1, u2, third, skip, limit, s, ds):
    """chi-square / 2 of (F + third) / (1 + third) for each draw not skipped: the flat line's, corrected over the bins in
    transit. The orbit is solved at a few knots across the transit and interpolated (cubic Hermite in x and y, which are
    smooth), and a bin with no contact in it is averaged over three instants: only the contacts need all ns.
    time is sorted by |t|, so the bins at the transit's centre come first, and a draw is dropped (inf) as soon as even a
    perfect fit to the rest of its window couldn't bring it under limit: a proof, not an estimate.
    The per-draw work stays in transit_change: numba hoists constant assignments out of a prange body."""
    n, flat = len(k), np.sum((data - 1) ** 2)
    out = np.full(n, 0.5 * flat / sigma ** 2)
    scale = 0.5 / sigma ** 2
    room = np.concatenate((np.zeros(1), np.cumsum((data - 1) ** 2)))  # the most the bins up to b can give back
    distance = np.abs(time)
    half_exp = exptime / 2
    edges = exptime * ((np.arange(1, ns + 1) - 0.5) / ns - 0.5)
    weights = np.array([5 / 18, 8 / 18, 5 / 18])
    for block in prange((n + BLOCK - 1) // BLOCK):
        xs, ys, vx, vy = np.empty(KNOTS_HIGH_E), np.empty(KNOTS_HIGH_E), np.empty(KNOTS_HIGH_E), np.empty(KNOTS_HIGH_E)
        front = np.empty(KNOTS_HIGH_E)
        for d in range(block * BLOCK, min((block + 1) * BLOCK, n)):
            if not skip[d]:
                out[d] += scale * transit_change(time, data, flat, scale, limit, room, distance, half_exp, edges, weights, k[d],
                                                 p[d], a[d], inc[d], ecc[d], w[d], u1[d], u2[d], third[d], xs, ys, vx, vy, front,
                                                 s, ds)
    return out


def scored(time, data, sigma, exptime, ns, k, p, a, inc, ecc, w, u1, u2, third, skip):
    """half_chi2 with the bins in order from the centre, and a limit from a pilot of the draws: MARGIN above the best of
    them. The pilot's best can only be worse than the true best, so the limit errs on the safe side."""
    order = np.argsort(np.abs(time))
    t, d = np.asarray(time, float)[order], np.asarray(data, float)[order]
    n = len(k)
    limit = np.inf
    if n > 4 * PILOT:
        pick = np.arange(0, n, n // PILOT)
        pilot = half_chi2(t, d, sigma, exptime, ns, k[pick], p[pick], a[pick], inc[pick], ecc[pick], w[pick], u1[pick], u2[pick],
                          third[pick], skip[pick], np.inf, S, DS)
        limit = np.min(pilot) + MARGIN
    return half_chi2(t, d, sigma, exptime, ns, k, p, a, inc, ecc, w, u1, u2, third, skip, limit, S, DS)


def plain(v, n):
    return np.broadcast_to(np.asarray(v, float), (n,)).copy()


def lnL_TP_p(time, flux, sigma, R_p, P_orb, inc, a, R_s, u1, u2, ecc, argp, companion_fluxratio,
             companion_is_host=False, exptime=0.00139, nsamples=20):
    """As TRICERATOPS's: half the chi-square of a transiting planet against the light curve, per draw."""
    n = len(R_p)
    fc = companion_fluxratio / (1 - companion_fluxratio)
    third = 1 / fc if companion_is_host else fc
    return scored(time, flux, float(sigma), float(exptime), int(nsamples), R_p * Rearth / (R_s * Rsun), plain(P_orb, n),
                  a / (R_s * Rsun), plain(inc * np.pi / 180, n), plain(ecc, n), (90 - argp) * np.pi / 180, plain(u1, n),
                  plain(u2, n), third, np.zeros(n, dtype=np.bool_))


def lnL_EB(time, flux, sigma, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp, companion_fluxratio,
           companion_is_host, exptime, nsamples, twin):
    n = len(R_EB)
    k, k_sec = R_EB / R_s, R_s / R_EB
    k = np.where(k - 1 < 1e-6, k * 0.999, k)  # sic: see the module's docstring
    k_sec = np.where(k_sec - 1 < 1e-6, k_sec * 0.999, k_sec)
    fc, fe = companion_fluxratio / (1 - companion_fluxratio), EB_fluxratio / (1 - EB_fluxratio)
    a_rs, p, incl = a / (R_s * Rsun), plain(P_orb, n), plain(inc * np.pi / 180, n)
    w, e, c1, c2 = (90 - argp) * np.pi / 180, plain(ecc, n), plain(u1, n), plain(u2, n)
    host, other = (fc, 1.0) if companion_is_host else (1.0, fc)  # the eclipsed star's light, and the third star's
    third = (fe + other) / host  # the primary: everything but the eclipsed star, over it
    if twin:
        seen = np.zeros(n, dtype=np.bool_)
    else:  # the secondary dims the EB, of light fe out of host + fe + other
        share = fe / (host + fe + other)
        seen = ~secondary_ok(k_sec, p, a_rs, incl, e, w + np.pi, c1, c2, np.zeros(n), share, float(sigma), SECONDARY_T, S, DS)
    out = scored(time, flux, float(sigma), float(exptime), int(nsamples), k, p, a_rs, incl, e, w, c1, c2, third, seen)
    out[seen] = np.inf
    return out


def lnL_EB_p(time, flux, sigma, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp, companion_fluxratio,
             companion_is_host=False, exptime=0.00139, nsamples=20):
    return lnL_EB(time, flux, sigma, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp, companion_fluxratio,
                  companion_is_host, exptime, nsamples, twin=False)


def lnL_EB_twin_p(time, flux, sigma, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp, companion_fluxratio,
                  companion_is_host=False, exptime=0.00139, nsamples=20):
    return lnL_EB(time, flux, sigma, R_EB, EB_fluxratio, P_orb, inc, a, R_s, u1, u2, ecc, argp, companion_fluxratio,
                  companion_is_host, exptime, nsamples, twin=True)


def ldc_of(teffs, loggs, table_teffs, table_loggs, table_u1, table_u2):
    """The limb darkening of each drawn star, looking up each distinct star once: stock loops over the million draws."""
    keys, first, back = np.unique(np.asarray(teffs, float) * 1e4 + loggs, return_index=True, return_inverse=True)
    u1, u2 = np.zeros(len(keys)), np.zeros(len(keys))
    for i, j in enumerate(first):
        mask = (table_teffs == teffs[j]) & (table_loggs == loggs[j])
        u1[i], u2[i] = table_u1[mask].item(), table_u2[mask].item()  # as stock: an error unless there's exactly one
    return u1[back], u2[back]


LOOP = re.compile(r"for i, \((\w+), (\w+)\) in enumerate\(\s*zip\((\w+), (\w+)\)\s*\):\s*"
                  r"mask = \((\w+) == \1\) & \((\w+) == \2\)\s*"
                  r"(\w+)\[i\], (\w+)\[i\] = (\w+)\[mask\]\.item\(\), (\w+)\[mask\]\.item\(\)")


def vectorised(ml, tt) -> list[str]:
    """Stock's per-draw limb-darkening loop, as one lookup: the scenarios' own source with that loop replaced, so it takes
    effect only on the code it was written for, and a new TRICERATOPS without the loop simply keeps its own."""
    done = []
    for name, f in list(vars(ml).items()):
        if name.startswith("lnZ_") and inspect.isfunction(f):
            source, n = LOOP.subn(r"\7, \8 = ldc_of(\3, \4, \5, \6, \9, \10)", inspect.getsource(f))
            if n:
                exec(compile(source, f"<tri_fast {name}>", "exec"), vars(ml))
                setattr(tt, name, getattr(ml, name))
                done.append(name)
    return done


def gaia_field(real, ra, dec, tic, kwargs):
    """Stock's Gaia query (its longest step, 52 s of 150 on one star), into the cache."""
    made = real(ra, dec, tic, **kwargs)
    if made is None:
        return None
    path = CACHE / f"gaia_{tic}.csv"
    shutil.move(made, path)
    return str(path)


def settle(target) -> None:
    """Wait for the star's Gaia query, which ran alongside the rest of the setup."""
    query = PENDING.pop(target.trilegal_fname, None)
    if query is not None and query.result() is None:
        target.trilegal_fname = None  # as stock: no field stars, so no blended-star scenarios


def install(cache: Path | None = None) -> None:
    """Point TRICERATOPS's scenarios at these likelihoods; calc_probs must run with parallel=True. cache: a folder to keep
    each star's setup in (pipeline/fpp.py), which also moves the Gaia query off the setup's path."""
    global Rsun, Rearth, CACHE
    import triceratops.marginal_likelihoods as ml
    import triceratops.triceratops as tt
    Rsun, Rearth = float(ml.Rsun), float(ml.Rearth)
    ml.lnL_TP_p, ml.lnL_EB_p, ml.lnL_EB_twin_p = lnL_TP_p, lnL_EB_p, lnL_EB_twin_p
    ml.ldc_of = ldc_of
    print(f"tri_fast: one lookup for the loop in {', '.join(vectorised(ml, tt)) or 'nothing'}", flush=True)
    if cache:
        CACHE = Path(cache)
        CACHE.mkdir(parents=True, exist_ok=True)
        real = tt.query_gaia_background

        def behind(ra, dec, tic, **kwargs):
            path = str(CACHE / f"gaia_{tic}.csv")
            if not Path(path).exists():
                PENDING[path] = POOL.submit(gaia_field, real, ra, dec, tic, kwargs)
            return path

        tt.query_gaia_background = behind


def golden_check() -> None:
    """Every draw of the golden calls, ours against stock (which is the saved answer), and how long each takes."""
    import time as clock

    import triceratops.likelihoods as ref
    global Rsun, Rearth
    Rsun, Rearth = float(ref.Rsun), float(ref.Rearth)
    folder = Path(__file__).resolve().parent.parent / "data" / "cache" / "fpp_golden"
    rest = ["P_orb", "inc", "a", "R_s", "u1", "u2", "ecc", "argp", "companion_fluxratio"]
    for name, ours in (("lnL_TP_p", lnL_TP_p), ("lnL_EB_p", lnL_EB_p), ("lnL_EB_twin_p", lnL_EB_twin_p)):
        z = np.load(next(folder.glob(f"*_{name}.npz")))
        draws = ["R_p"] if name == "lnL_TP_p" else ["R_EB", "EB_fluxratio"]
        host = bool(z["companion_is_host"]) if "companion_is_host" in z.files else False
        args = {k: z[k].copy() for k in draws + rest}
        args["inc"] = args["inc"] * 180 / np.pi  # fpp_golden recorded it after stock's in-place conversion
        call = lambda f: f(z["time"], z["flux"], float(z["sigma"]), *[args[k].copy() for k in draws + rest], host,  # noqa: E731, B023
                           float(z["exptime"]), int(z["nsamples"]))
        ours(z["time"][:4], z["flux"][:4], 1.0, *[args[k][:4].copy() for k in draws + rest], host, 0.007, 2)  # compile
        t0 = clock.time()
        lnl = call(ours)
        t1 = clock.time()
        call(getattr(ref, name))
        call(getattr(ref, name))  # the second, for a time without the first call's compile
        t2 = clock.time()
        call(getattr(ref, name))
        t3 = clock.time()
        want = z["lnL"]
        fin = np.isfinite(lnl) & np.isfinite(want)
        d = np.abs(lnl[fin] - want[fin])
        best = want[fin] < want[fin].min() + 30
        e_low = z["ecc"][fin] < 0.5
        print(f"{name}: {t1 - t0:.2f} s vs stock {t3 - t2:.2f} s ({(t3 - t2) / (t1 - t0):.1f}x); same inf {(np.isfinite(lnl) == np.isfinite(want)).mean():.5f}; "
              f"lnL gap median {np.median(d):.1e}, within 30 of best: max {d[best].max():.1e} (at e<0.5 {d[best & e_low].max():.1e}, n {best.sum()})")
        lz = np.logaddexp.reduce(-lnl[np.isfinite(lnl)]), np.logaddexp.reduce(-want[np.isfinite(want)])
        print(f"   log-sum of the evidence weights: ours {lz[0]:.6f}, stock {lz[1]:.6f}, gap {lz[0] - lz[1]:+.2e}")


if __name__ == "__main__":
    golden_check()
