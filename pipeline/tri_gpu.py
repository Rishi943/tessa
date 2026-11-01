"""pipeline/tri_fast.py's chi-square on a GPU (CuPy), to measure what a GPU adds.

The same physics as tri_fast.transit_change, as arrays: each batch of draws gets its orbit solved at knots, then a flat list
of (draw, bin) pairs is sorted into off the disc, no contact (3 instants) and a contact (all ns instants). Bins near the
centre run first, and draws that provably can't come within MARGIN of the best are dropped before the rest. Values near
1 are carried as 1 - flux, so single precision loses nothing that matters. High-eccentricity draws stay on the CPU kernel,
which adapts its window and knots to them.

Run: python -m pipeline.tri_gpu      golden calls: parity, and speed against stock
"""
import numpy as np

try:
    import cupy as xp
except ImportError:
    xp = np

from . import tri_fast as cpu

FLOAT = xp.float32
BATCH = 40_000  # draws at a time
KNOTS = cpu.KNOTS
GAUSS = np.array([5 / 18, 8 / 18, 5 / 18])
to_cpu = (lambda a: a) if xp is np else xp.asnumpy  # noqa: E731
_x, _w = np.polynomial.legendre.leggauss(cpu.NODES)
_phi = (_x + 1) * np.pi / 2
S = xp.asarray((1 - np.cos(_phi)) / 2, dtype=FLOAT)
DS = xp.asarray(np.sin(_phi) / 2 * _w * np.pi / 2, dtype=FLOAT)


def arr(v):
    return xp.asarray(np.asarray(v, float), dtype=FLOAT)


def blocked(z, k, u1, u2):
    """tri_fast.blocked, on arrays."""
    c0, c1, c2 = 1 - u1 - u2, u1 + 2 * u2, -u2
    total = 1 - u1 / 3 - u2 / 6
    k2, z2 = k * k, z * z
    out = xp.zeros_like(z)
    small = (k < cpu.SMALL_K) & (z + k < cpu.CLEAR_OF_LIMB)
    if bool(small.any()):
        zs, ks = z2[small], k2[small]
        m2 = 1 - zs
        m = xp.sqrt(m2)
        lap = -(2 - zs) / (m * m2)
        lap2 = (zs * zs - 8 * zs - 8) / (m * m2 * m2 * m2)
        mean_mu = m + ks / 8 * lap + ks * ks / 192 * lap2
        out[small] = ks * (c0[small] + c1[small] * mean_mu + c2[small] * (m2 - ks / 2)) / total[small]
    rest = ~small
    if bool(rest.any()):
        z_, k_, c0_, c1_, c2_ = xp.maximum(z[rest], 1e-6), k[rest], c0[rest], c1[rest], c2[rest]
        inner = xp.clip(k_ - z_, 0, 1)
        inner2 = inner * inner
        mu_in = xp.sqrt(xp.clip(1 - inner2, 0, None))
        disc = c0_ * inner2 + c1_ * 2 / 3 * (1 - mu_in ** 3) + c2_ * (inner2 - inner2 * inner2 / 2)
        lo, hi = xp.abs(z_ - k_), xp.minimum(z_ + k_, 1)
        width = xp.clip(hi - lo, 0, None)
        r = xp.maximum(lo[:, None] + width[:, None] * S, 1e-6)
        r2 = r * r
        mu = xp.sqrt(xp.clip(1 - r2, 0, None))
        light = c0_[:, None] + c1_[:, None] * mu + c2_[:, None] * mu * mu
        half = xp.arccos(xp.clip((r2 + (z_ * z_ - k_ * k_)[:, None]) / (2 * r * z_[:, None]), -1, 1))
        band = width * xp.sum(light * 2 * r * half * DS, axis=1) / np.pi
        out[rest] = (disc + band) / total[rest]
    return out


class Orbits:
    """A batch of draws, solved at the knots."""

    def __init__(self, k, p, a, inc, ecc, w, u1, u2, third):
        self.k, self.u1, self.u2, self.third = k, u1, u2, third
        f = np.pi / 2 - w
        e0 = 2 * xp.arctan2(xp.sqrt(1 - ecc) * xp.sin(f / 2), xp.sqrt(1 + ecc) * xp.cos(f / 2))
        self.m0 = e0 - ecc * xp.sin(e0)
        self.rate, self.a, self.ecc, self.w = 2 * np.pi / p, a, ecc, w
        self.sin_i, self.cos_i = xp.sin(inc), xp.cos(inc)
        self.reach = (1 + k) ** 2
        self.inner = xp.where(k < 1, (1 - k) ** 2, -1.0)
        limit = p / 4
        window = xp.minimum(1.6 * (1 + k) / (self.rate * a * (1 + ecc * xp.sin(w)) / xp.sqrt(1 - ecc * ecc)), limit)
        for _ in range(4):  # as tri_fast: the speed changes during the transit
            xa, ya, *_ = self.solve(window[:, None])
            xb, yb, *_ = self.solve(-window[:, None])
            short = ((xa * xa + ya * ya < self.reach[:, None]) | (xb * xb + yb * yb < self.reach[:, None]))[:, 0]
            window = xp.where(short & (window < limit), xp.minimum(window * 1.5, limit), window)
        self.window, self.h = window, 2 * window / (KNOTS - 1)
        knots = -window[:, None] + self.h[:, None] * xp.arange(KNOTS, dtype=FLOAT)
        self.xs, self.ys, self.vx, self.vy, self.front = self.solve(knots)

    def solve(self, t):
        m0, rate, e, a = self.m0[:, None], self.rate[:, None], self.ecc[:, None], self.a[:, None]
        mean = rate * t + m0
        ea = mean + 0.85 * e * xp.sign(xp.sin(mean))
        for _ in range(6):
            ea = ea - (ea - e * xp.sin(ea) - mean) / (1 - e * xp.cos(ea))
        sin_e, cos_e = xp.sin(ea), xp.cos(ea)
        root = xp.sqrt(1 - e * e)
        away = 1 - e * cos_e
        theta = self.w[:, None] + xp.arctan2(root * sin_e, cos_e - e)
        sin_t, cos_t = xp.sin(theta), xp.cos(theta)
        r = a * away
        r_dot = a * e * sin_e * rate / away
        nu_dot = rate * root / (away * away)
        cos_i = self.cos_i[:, None]
        return (r * cos_t, r * sin_t * cos_i, r_dot * cos_t - r * sin_t * nu_dot, (r_dot * sin_t + r * cos_t * nu_dot) * cos_i,
                r * sin_t * self.sin_i[:, None] > 0)

    def position(self, t, rows):
        h = self.h[rows]
        v = xp.clip((t + self.window[rows]) / h, 0, KNOTS - 1 - 1e-3)
        j = v.astype(xp.int32)
        f = v - j
        g = 1 - f
        c0, c1, c2, c3 = (1 + 2 * f) * g * g, f * g * g * h, f * f * (3 - 2 * f), -f * f * g * h
        x = c0 * self.xs[rows, j] + c1 * self.vx[rows, j] + c2 * self.xs[rows, j + 1] + c3 * self.vx[rows, j + 1]
        y = c0 * self.ys[rows, j] + c1 * self.vy[rows, j] + c2 * self.ys[rows, j + 1] + c3 * self.vy[rows, j + 1]
        return x * x + y * y, x, j

    def dimming(self, t, rows):
        """1 - flux at t for each pair."""
        z2, _, j = self.position(t, rows)
        hit = (xp.abs(t) < self.window[rows]) & (z2 < self.reach[rows]) & (self.front[rows, j] | self.front[rows, j + 1])
        out = xp.zeros_like(t)
        if bool(hit.any()):
            r = rows[hit]
            out[hit] = blocked(xp.sqrt(z2[hit]), self.k[r], self.u1[r], self.u2[r])
        return out


def region(z2, inner, reach):
    return (z2 >= inner).astype(xp.int8) + (z2 >= reach).astype(xp.int8)


def change_over(o, t, d, half_exp, edges, draws):
    """What the bins t add to chi-square, for the draws (positions in the batch), as a double-precision array over them."""
    m = int(o.k.shape[0])
    near = xp.abs(t)[None, :] <= (o.window[draws] + half_exp)[:, None]
    picked, cols = xp.nonzero(near)
    rows = draws[picked]
    tb = t[cols]
    z_a, x_a, _ = o.position(tb - half_exp, rows)
    z_b, x_b, _ = o.position(tb + half_exp, rows)
    same = x_a * x_b > 0
    ra, rb = region(z_a, o.inner[rows], o.reach[rows]), region(z_b, o.inner[rows], o.reach[rows])
    off = (ra == 2) & (rb == 2) & same
    smooth = (ra == rb) & same & ~off
    edge = ~off & ~smooth
    out = xp.zeros(m, dtype=xp.float64)
    for keep, instants, share in ((smooth, (-cpu.ROOT_06 * half_exp, 0.0, cpu.ROOT_06 * half_exp), GAUSS),
                                  (edge, edges, np.full(len(edges), 1 / len(edges)))):
        if not bool(keep.any()):
            continue
        r, ts = rows[keep], tb[keep]
        dim = xp.zeros_like(ts)
        for instant, weight in zip(instants, share):
            dim += float(weight) * o.dimming(ts + float(instant), r)
        delta = dim / (1 + o.third[r])  # 1 - model
        change = delta * (2 * d[cols[keep]] + delta)  # (data - model)^2 - (data - 1)^2, with d = data - 1
        out += xp.bincount(r, weights=change.astype(xp.float64), minlength=m)
    return out


def half_chi2(t, d, room, flat, scale, half_exp, edges, k, p, a, inc, e, w, u1, u2, third, limit):
    """The batches: the central bins, then the rest for the draws still in reach of limit."""
    n, out = len(k), np.full(len(k), flat * scale)
    middle = min(max(len(t) // 6, 8), len(t))
    for i in range(0, n, BATCH):
        j = min(i + BATCH, n)
        o = Orbits(*[arr(v[i:j]) for v in (k, p, a, inc, e, w, u1, u2, third)])
        every = xp.arange(j - i)
        change = change_over(o, t[:middle], d[:middle], half_exp, edges, every)
        last = xp.searchsorted(xp.abs(t), o.window + half_exp, side="right")
        left = xp.maximum(room[last] - room[middle], 0)  # the most the rest of each draw's window can give back
        alive = xp.nonzero((flat + change - left) * scale <= limit)[0]
        if int(alive.size) and middle < len(t):
            change = change + change_over(o, t[middle:], d[middle:], half_exp, edges, alive)
        lnl = to_cpu(change) * scale + flat * scale
        dead = to_cpu((flat + change - left) * scale > limit)
        lnl[dead] = np.inf
        out[i:j] = lnl
    return out


def scored(time, data, sigma, exptime, ns, k, p, a, inc, ecc, w, u1, u2, third, skip):
    """As tri_fast.scored, on the GPU."""
    order = np.argsort(np.abs(time))
    tn, dn = np.asarray(time, float)[order], np.asarray(data, float)[order]
    flat, scale = float(np.sum((dn - 1) ** 2)), 0.5 / sigma ** 2
    t, d = arr(tn), arr(dn - 1)
    room = xp.asarray(np.concatenate(([0.0], np.cumsum((dn - 1) ** 2))))
    half_exp = float(exptime / 2)
    edges = exptime * ((np.arange(1, ns + 1) - 0.5) / ns - 0.5)
    n = len(k)
    out = np.full(n, flat * scale)
    gpu = np.flatnonzero(~skip & (ecc < cpu.HIGH_E))
    high = np.flatnonzero(~skip & (ecc >= cpu.HIGH_E))
    pick = lambda idx: [v[idx] for v in (k, p, a, inc, ecc, w, u1, u2, third)]  # noqa: E731
    limit = np.inf
    if len(gpu) > 4 * cpu.PILOT:
        pilot = half_chi2(t, d, room, flat, scale, half_exp, edges, *pick(gpu[:: len(gpu) // cpu.PILOT]), np.inf)
        limit = float(np.min(pilot)) + cpu.MARGIN
    if len(gpu):
        out[gpu] = half_chi2(t, d, room, flat, scale, half_exp, edges, *pick(gpu), limit)
    if len(high):
        out[high] = cpu.half_chi2(tn, dn, float(sigma), float(exptime), ns, *pick(high), np.zeros(len(high), dtype=np.bool_), limit,
                                  cpu.S, cpu.DS)
    return out


def install(cache=None) -> None:
    cpu.install(cache)
    cpu.scored = scored


if __name__ == "__main__":
    print(f"array library: {xp.__name__}", flush=True)
    cpu.scored = scored
    cpu.golden_check()
