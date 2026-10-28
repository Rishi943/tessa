"""Transit model fit and Gaia lookups shared by the vetting tools, the candidate package and the
fine-tuning clues.

transit_fit: a limb-darkened transit (numpy only: exact uniform-disk overlap, limb darkening at the planet's
position, exposure smearing) fitted by least squares: radius ratio, impact parameter, a/R*, the star density
the orbit implies, and how much worse the star's own density fits (the star-density test).
gaia_extra / gaia_ebs: Gaia DR3 binarity signs and FLAME/GSP-Phot radii, and the eclipsing-binary list.
"""
import math
import time

import numpy as np

U1, U2 = 0.40, 0.20  # quadratic limb darkening, TESS band, roughly solar; the fit is a feature, not a measurement
SUPERSAMPLE = 5
G_CGS = 6.674e-8


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {msg}", flush=True)


def retry(fn, attempts=4):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # the Gaia archive returns sporadic 500s
            if attempt == attempts - 1:
                raise
            log(f"{e.__class__.__name__}, retrying in 20 s")
            time.sleep(20)


def overlap(z: np.ndarray, k: float) -> np.ndarray:
    """Fraction of the stellar disk a planet of radius k covers at separation z (uniform disk)."""
    out = np.zeros_like(z)
    inside = z <= 1 - k
    out[inside] = k * k
    if k > 1:
        out[z <= k - 1] = 1.0
    part = (z > abs(1 - k)) & (z < 1 + k)
    zp = z[part]
    k0 = np.arccos(np.clip((k * k + zp * zp - 1) / (2 * k * zp), -1, 1))
    k1 = np.arccos(np.clip((1 - k * k + zp * zp) / (2 * zp), -1, 1))
    root = np.sqrt(np.clip(4 * zp * zp - (1 + zp * zp - k * k) ** 2, 0, None)) / 2
    out[part] = (k * k * k0 + k1 - root) / np.pi
    return out


def model(t: np.ndarray, span: np.ndarray, p: float, t0: float, k: float, b: float, ars: float) -> np.ndarray:
    """Relative flux (0 = out of transit), averaged over each point's exposure."""
    offsets = (np.arange(SUPERSAMPLE) + 0.5) / SUPERSAMPLE - 0.5
    tt = t[:, None] + span[:, None] * offsets[None, :]
    phi = 2 * np.pi * (tt - t0) / p
    z = np.hypot(ars * np.sin(phi), b * np.cos(phi))
    z = np.where(np.cos(phi) > 0, z, 10.0)  # behind the star
    r = np.clip(z, 0, 1)
    mu = np.sqrt(1 - r * r)
    intensity = (1 - U1 * (1 - mu) - U2 * (1 - mu) ** 2) / (1 - U1 / 3 - U2 / 6)
    return -(overlap(z.ravel(), k).reshape(z.shape) * intensity).mean(axis=1)


def transit_fit(lc: dict, sig: dict, star: dict) -> dict:
    from scipy.optimize import least_squares
    p, t0, dur = sig["period"], sig["t0"], sig["dur"]
    phase = (lc["t"] - t0 + p / 2) % p - p / 2
    near = np.abs(phase) < 2 * dur
    if near.sum() < 20:
        return {"status": "too few points"}
    t, y, sw, span = lc["t"][near], lc["y"][near], np.sqrt(lc["w"][near]), lc["span"][near]
    depth = max(sig["depth_ppm"] * 1e-6, 1e-5)

    def resid(x):
        dt0, logk, b, logars = x
        return sw * (y - model(t, span, p, t0 + dt0, math.exp(logk), b, math.exp(logars)))

    best = None
    for b0 in (0.1, 0.5, 0.85):
        k0 = min(math.sqrt(depth) * 1.1, 0.9)
        ars0 = max(p / (math.pi * dur) * math.sqrt(max((1 + k0) ** 2 - b0 * b0, 0.05)), 1.6)
        try:
            r = least_squares(resid, [0.0, math.log(k0), b0, math.log(ars0)],
                              bounds=([-dur / 2, math.log(0.003), 0, math.log(1.5)],
                                      [dur / 2, math.log(1.0), 1.5, math.log(300)]), max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        return {"status": "fit failed"}
    dt0, logk, b, logars = best.x
    k, ars = math.exp(logk), math.exp(logars)
    rho_fit = 3 * math.pi * ars ** 3 / (G_CGS * (p * 86400) ** 2)
    rho_star = 1.41 * star["mass"] / star["rad"] ** 3 if star.get("mass") and star.get("rad") else None
    t14 = p / math.pi * math.asin(min(1.0, math.sqrt(max((1 + k) ** 2 - b * b, 0)) / ars)) * 24 if ars > b else None
    chi2_flat = float(np.sum((sw * y) ** 2))
    out = {"status": "ok", "k": k, "b": float(b), "ars": ars, "dt0_min": float(dt0) * 1440,
           "rho_fit": rho_fit, "rho_star": rho_star, "rho_ratio": rho_fit / rho_star if rho_star else None,
           "t14_h": t14, "grazing": bool(b > 1 - k), "delta_chi2": chi2_flat - 2 * float(best.cost),
           "chi2_red": 2 * float(best.cost) / max(int(near.sum()) - 4, 1)}
    # The same fit with a/R* fixed by the star's own density: on faint data b and a/R* trade off, so the
    # free fit's density is loose, but how much worse the star's density fits is a clean test of
    # "is this dip the right shape for a planet on this star?" (blends and binaries usually fail it)
    if rho_star:
        ars_star = (rho_star * G_CGS * (p * 86400) ** 2 / (3 * math.pi)) ** (1 / 3)
        if ars_star > 1.2:
            def resid_star(x):
                return sw * (y - model(t, span, p, t0 + x[0], math.exp(x[1]), x[2], ars_star))
            fixed = None
            for b0 in (0.1, 0.5, 0.85):
                try:
                    r = least_squares(resid_star, [dt0, logk, b0], bounds=([-dur / 2, math.log(0.003), 0],
                                      [dur / 2, math.log(1.0), 1.5]), max_nfev=300)
                except Exception:
                    continue
                if fixed is None or r.cost < fixed.cost:
                    fixed = r
            if fixed is not None:
                out.update(ars_star=ars_star, b_star=float(fixed.x[2]), k_star=math.exp(fixed.x[1]),
                           star_density_penalty=2 * float(fixed.cost - best.cost))
    return out



def gaia_extra(source_ids: list[int]) -> dict[str, dict]:
    from astroquery.gaia import Gaia
    cols = ("g.source_id AS sid, g.ra, g.dec, g.radial_velocity_error, g.rv_nb_transits, g.rv_chisq_pvalue, "
            "g.rv_renormalised_gof, g.rv_amplitude_robust, g.ipd_frac_multi_peak, g.ipd_gof_harmonic_amplitude, "
            "g.astrometric_excess_noise_sig, a.radius_flame, a.mass_flame, a.radius_gspphot, a.logg_gspphot")
    out = {}
    for i in range(0, len(source_ids), 500):
        ids = ",".join(str(s) for s in source_ids[i:i + 500])
        q = (f"SELECT {cols} FROM gaiadr3.gaia_source AS g LEFT JOIN gaiadr3.astrophysical_parameters AS a "
             f"ON a.source_id = g.source_id WHERE g.source_id IN ({ids})")
        df = retry(lambda: Gaia.launch_job(q, verbose=False).get_results().to_pandas())
        df.columns = [c.lower() for c in df.columns]
        sids = df.pop("sid").astype("int64").tolist()  # 19-digit IDs: never through float
        df = df[[c for c in df.columns if not c.startswith("source_id")]]
        for sid, r in zip(sids, df.to_dict("records")):
            out[str(sid)] = {k: (None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v))
                             for k, v in r.items()}
        log(f"Gaia extras {min(i + 500, len(source_ids))}/{len(source_ids)}")
    return out


def gaia_ebs(source_ids: list[int]) -> dict[str, float]:
    """Gaia DR3 eclipsing-binary periods (days) for these sources."""
    from astroquery.gaia import Gaia
    out = {}
    for i in range(0, len(source_ids), 1000):
        ids = ",".join(str(s) for s in source_ids[i:i + 1000])
        df = retry(lambda: Gaia.launch_job(f"SELECT source_id, frequency FROM gaiadr3.vari_eclipsing_binary "
                                           f"WHERE source_id IN ({ids})", verbose=False).get_results().to_pandas())
        df.columns = [c.lower() for c in df.columns]
        out.update({str(int(s)): 1 / float(f) for s, f in zip(df["source_id"], df["frequency"]) if f})
    log(f"Gaia eclipsing binaries: {len(out)} of {len(source_ids)} sources")
    return out


def eb_match(period: float, eb_period: float | None) -> float | None:
    """How close the EB period is to this period or a simple harmonic (0 = exact); None if not an EB."""
    if not eb_period:
        return None
    return min(abs(period * h / eb_period - 1) for h in (0.5, 1, 2, 3, 1 / 3))


# --- per TOI -----------------------------------------------------------------------------------

