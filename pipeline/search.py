"""Search light curves for transits: detrend, box least squares (BLS) on the GPU, gate 1.

Gate 1 asks whether a star's best periodic dip is real. Planets don't make periodic brightenings,
so the same search on the light curve flipped upside down (the best "bump") shows how far noise
and systematics alone reach in this star. The dip must beat that by MARGIN, clear an SNR floor,
and have at least MIN_TRANSITS transits with data, none of them carrying most of the signal.
Both sides go through the identical procedure, so the comparison is fair.

The floor is measured, not chosen: calibrate() takes the lowest floor at which the same gate run
on the flipped light curves passes at most FALSE_PASS_RATE of the stars. Those flipped passes are
the estimate of how many real passes are false alarms. It runs when a survey finishes.

Self-contained (no package imports) so it can run as a Kaggle job on the fetch output:
  python search.py --lc /kaggle/input --stars survey_stars.txt --skip holdout.txt --out /kaggle/working --gpus 2
Without CuPy it runs the NumPy reference, which is fine for a star or two and slow beyond that.

Output: results.csv (one row per star: best dip, best bump, gate 1 verdict) and peaks.csv
(the top periodogram peaks on each side with their per-transit checks).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

# Detrending
WINDOW_D = 0.5  # running-median window; >= 3x the longest transit searched
GAP_D = 0.5  # a gap this long starts a new segment
FLARE_SIGMA = 3.5  # brightenings above this are flares (M dwarfs flare a lot) and are dropped,
FLARE_PAD_D = (10 / 1440, 30 / 1440)  # with this much before and after for the decay
BIN_D = 10 / 1440  # 200-s sectors are binned to 10 min; transits are >= 30 min

# Search grid
P_MIN, P_MAX = 0.5, 15.0
PHASE_BIN_D = 0.25 / 24
DUR_BINS = np.array([2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24], np.int32)  # 0.5-6 h in phase bins
DUR_RANGE = (0.3, 1.5)  # allowed durations, as fractions of a central transit's at that period
OVERSAMPLE = 3  # phase drift over the baseline is at most 1/OVERSAMPLE of the transit duration
MIN_IN_POINTS = 4
NBMAX = 2048
BLOCK = 256

# Peaks and gate 1
TOP_PEAKS = 10
PEAK_SEPARATION = 0.002  # peaks closer than this in relative period are the same peak
MIN_TRANSITS = 3
MIN_COVERAGE = 0.5  # a transit counts if data covers at least half of it
MAX_EVENT_SHARE = 0.5
MARGIN = 1.2
SNR_FLOOR = 10.0  # provisional, for per-star logs; calibrate() measures the real one
FALSE_PASS_RATE = 0.01

RSUN_PER_AU = 215.03
T0 = time.time()

KERNEL = r"""
#define NBMAX %(nbmax)d
#define BLOCK %(block)d
// One block per trial frequency: fold into phase bins in shared memory, then slide boxes of the
// allowed durations over every start bin (wrapping) and keep the best dip and the best bump.
extern "C" __global__ void bls(
    const double* __restrict__ t, const float* __restrict__ y, const float* __restrict__ w, const int n,
    const double* __restrict__ freq, const int* __restrict__ nbin,
    const int* __restrict__ klo, const int* __restrict__ khi,
    const int* __restrict__ kgrid, const int nk, const float wtot, const int min_pts,
    float* __restrict__ stat, int* __restrict__ code)
{
    __shared__ float sw[NBMAX];
    __shared__ float sy[NBMAX];
    __shared__ int sc[NBMAX];
    __shared__ float rstat[2][BLOCK];
    __shared__ int rcode[2][BLOCK];
    const int f = blockIdx.x;
    const int nb = nbin[f];
    for (int b = threadIdx.x; b < nb; b += BLOCK) { sw[b] = 0.f; sy[b] = 0.f; sc[b] = 0; }
    __syncthreads();

    const double fr = freq[f];
    for (int j = threadIdx.x; j < n; j += BLOCK) {
        double ph = t[j] * fr;  // double: t * f reaches ~5000 cycles over a 7-year baseline
        ph -= floor(ph);
        int b = (int)(ph * nb);
        if (b >= nb) b = nb - 1;
        atomicAdd(&sw[b], w[j]);
        atomicAdd(&sy[b], w[j] * y[j]);
        atomicAdd(&sc[b], 1);
    }
    __syncthreads();

    float best0 = 0.f, best1 = 0.f;
    int code0 = -1, code1 = -1;
    const int lo = klo[f], hi = khi[f];
    for (int i = threadIdx.x; i < nb; i += BLOCK) {
        float r = 0.f, s = 0.f;
        int c = 0, j = 0;
        for (int q = 0; q < nk; q++) {
            const int k = kgrid[q];
            if (k > hi) break;
            for (; j < k; j++) {
                int b = i + j;
                if (b >= nb) b -= nb;
                r += sw[b]; s += sy[b]; c += sc[b];
            }
            if (k < lo || c < min_pts || r <= 0.f || r >= wtot) continue;
            const float st = s * s * wtot / (r * (wtot - r));
            if (s < 0.f) { if (st > best0) { best0 = st; code0 = q * NBMAX + i; } }
            else if (st > best1) { best1 = st; code1 = q * NBMAX + i; }
        }
    }
    rstat[0][threadIdx.x] = best0; rcode[0][threadIdx.x] = code0;
    rstat[1][threadIdx.x] = best1; rcode[1][threadIdx.x] = code1;
    __syncthreads();
    for (int h = BLOCK / 2; h > 0; h >>= 1) {
        if (threadIdx.x < h) {
            for (int side = 0; side < 2; side++) {
                if (rstat[side][threadIdx.x + h] > rstat[side][threadIdx.x]) {
                    rstat[side][threadIdx.x] = rstat[side][threadIdx.x + h];
                    rcode[side][threadIdx.x] = rcode[side][threadIdx.x + h];
                }
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        stat[2 * f] = rstat[0][0]; stat[2 * f + 1] = rstat[1][0];
        code[2 * f] = rcode[0][0]; code[2 * f + 1] = rcode[1][0];
    }
}
""" % {"nbmax": NBMAX, "block": BLOCK}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min  {msg}", flush=True)


# ---------------------------------------------------------------- detrending

def detrend(t: np.ndarray, flux: np.ndarray, sector: np.ndarray) -> dict[str, np.ndarray]:
    """Running-median detrend per segment, flare removal, then 10-minute bins.

    Returns time, relative flux (0 = median), inverse-variance weights from each sector's own
    scatter (FFI pipeline errors are not trusted), and the time each point covers."""
    order = np.argsort(t, kind="stable")
    t, flux, sector = t[order], flux[order].astype(np.float64), sector[order]

    cadence = np.empty_like(t)
    for s in np.unique(sector):
        m = sector == s
        cadence[m] = np.median(np.diff(t[m])) if m.sum() > 1 else BIN_D

    starts = np.flatnonzero((np.diff(t) > GAP_D) | (np.diff(sector) != 0)) + 1
    rel = np.empty_like(flux)
    for a, b in zip(np.r_[0, starts], np.r_[starts, len(t)]):
        size = max(3, int(round(WINDOW_D / cadence[a])) | 1)
        trend = median_filter(flux[a:b], size=size, mode="nearest")
        rel[a:b] = flux[a:b] / trend - 1

    noise = np.empty_like(rel)
    for s in np.unique(sector):
        m = sector == s
        noise[m] = 1.4826 * np.median(np.abs(rel[m] - np.median(rel[m])))
    ok = np.isfinite(rel) & (noise > 0)

    flare = ok & (rel > FLARE_SIGMA * noise)
    if flare.any():
        edges = np.zeros(len(t) + 1, np.int64)
        np.add.at(edges, np.searchsorted(t, t[flare] - FLARE_PAD_D[0], "left"), 1)
        np.add.at(edges, np.searchsorted(t, t[flare] + FLARE_PAD_D[1], "right"), -1)
        ok &= np.cumsum(edges)[:-1] == 0

    t, rel, noise, cadence, sector = t[ok], rel[ok], noise[ok], cadence[ok], sector[ok]
    w = 1 / noise**2
    # Bin within each sector; 30- and 10-minute sectors stay at one point per bin.
    key = sector.astype(np.int64) * 10**8 + np.floor((t - t.min()) / BIN_D).astype(np.int64)
    keys, idx = np.unique(key, return_inverse=True)
    sw = np.bincount(idx, w)
    out = {
        "t": np.bincount(idx, w * t) / sw,
        "y": np.bincount(idx, w * rel) / sw,
        "w": sw,
        "span": np.bincount(idx, cadence),
        "sector": (keys // 10**8).astype(np.int16),
        "noise_ppm": float(np.median(noise)) * 1e6,
        "n_flare": int(flare.sum()),
    }
    out["y"] -= np.sum(out["w"] * out["y"]) / out["w"].sum()
    return out


# ---------------------------------------------------------------- search grid

def central_duration(period: np.ndarray, rad: float, mass: float) -> np.ndarray:
    """Duration (days) of a central transit on a circular orbit."""
    a = RSUN_PER_AU * (mass * (period / 365.25) ** 2) ** (1 / 3)
    return period / np.pi * np.arcsin(np.minimum(1.0, rad / a))


def grid(baseline: float, rad: float, mass: float, p_min: float = P_MIN, p_max: float = P_MAX):
    """Trial frequencies, phase bins per frequency and the allowed duration range in bins.

    Central durations scale as P^(1/3), so the step that keeps phase drift over the baseline
    below duration/OVERSAMPLE is uniform in f^(1/3)."""
    p_max = min(p_max, baseline / (MIN_TRANSITS - 1))
    c = float(central_duration(np.array(1.0), rad, mass))
    du = c / (3 * OVERSAMPLE * baseline)
    u = np.arange((1 / p_max) ** (1 / 3), (1 / p_min) ** (1 / 3), du)
    freq = u**3
    period = 1 / freq
    nbin = np.ceil(period / PHASE_BIN_D).astype(np.int32)
    if nbin.max() > NBMAX:
        raise ValueError(f"P_MAX needs {nbin.max()} phase bins; NBMAX is {NBMAX}")
    dur_bins = central_duration(period, rad, mass) / (period / nbin)
    khi = np.clip(np.ceil(DUR_RANGE[1] * dur_bins), DUR_BINS[0], np.minimum(DUR_BINS[-1], nbin // 4))
    klo = np.minimum(np.floor(DUR_RANGE[0] * dur_bins), khi)
    return freq, nbin, klo.astype(np.int32), khi.astype(np.int32)


# ---------------------------------------------------------------- BLS

def bls_numpy(t, y, w, freq, nbin, klo, khi, min_pts=MIN_IN_POINTS):
    """Reference implementation of the kernel. t is relative to the first point."""
    wtot = w.sum()
    stat = np.zeros((len(freq), 2), np.float32)
    code = np.full((len(freq), 2), -1, np.int32)
    for m, fr in enumerate(freq):
        nb = int(nbin[m])
        ph = t * fr
        b = np.minimum(((ph - np.floor(ph)) * nb).astype(np.int64), nb - 1)
        sw = np.bincount(b, w, nb)
        sy = np.bincount(b, w * y, nb)
        sc = np.bincount(b, minlength=nb)
        kmax = int(khi[m])
        cw, cy, cc = (np.r_[0, np.cumsum(np.r_[a, a[:kmax]])] for a in (sw, sy, sc))
        for q, k in enumerate(DUR_BINS):
            if k > kmax:
                break
            if k < klo[m]:
                continue
            r, s, c = cw[k:k + nb] - cw[:nb], cy[k:k + nb] - cy[:nb], cc[k:k + nb] - cc[:nb]
            ok = (c >= min_pts) & (r > 0) & (r < wtot)
            st = np.where(ok, s * s * wtot / np.where(ok, r * (wtot - r), 1), 0)
            for side, mask in ((0, s < 0), (1, s >= 0)):
                cand = np.where(mask, st, 0)
                i = int(np.argmax(cand))
                if cand[i] > stat[m, side]:
                    stat[m, side], code[m, side] = cand[i], q * NBMAX + i
    return stat, code


class GpuBLS:
    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.kernel = cp.RawKernel(KERNEL, "bls")

    def __call__(self, t, y, w, freq, nbin, klo, khi, min_pts=MIN_IN_POINTS):
        cp = self.cp
        nf = len(freq)
        stat = cp.zeros(2 * nf, cp.float32)
        code = cp.full(2 * nf, -1, cp.int32)
        args = (cp.asarray(t, cp.float64), cp.asarray(y, cp.float32), cp.asarray(w, cp.float32), np.int32(len(t)),
                cp.asarray(freq, cp.float64), cp.asarray(nbin), cp.asarray(klo), cp.asarray(khi),
                cp.asarray(DUR_BINS), np.int32(len(DUR_BINS)), np.float32(w.sum()), np.int32(min_pts),
                stat, code)
        self.kernel((nf,), (BLOCK,), args)
        return cp.asnumpy(stat).reshape(nf, 2), cp.asnumpy(code).reshape(nf, 2)


# ---------------------------------------------------------------- peaks and gate 1

def check_peak(lc: dict, period: float, t0: float, dur: float, sign: int) -> dict:
    """Depth, SNR and per-transit evidence for a box; sign -1 for a dip, +1 for a bump."""
    t, y, w = lc["t"], lc["y"], lc["w"]
    epoch = np.round((t - t0) / period)
    inside = np.abs(t - t0 - epoch * period) < dur / 2
    W, r = w.sum(), w[inside].sum()
    s = np.sum(w[inside] * y[inside])
    amp = sign * s * W / (r * (W - r)) if 0 < r < W else 0.0
    snr = amp * np.sqrt(r * (W - r) / W) if 0 < r < W else 0.0

    ep, inv = np.unique(epoch[inside], return_inverse=True)
    r_e = np.bincount(inv, w[inside])
    s_e = np.bincount(inv, w[inside] * y[inside])
    cover = np.bincount(inv, lc["span"][inside]) / dur
    snr_e = np.maximum(sign * s_e / np.sqrt(r_e), 0) if len(ep) else np.zeros(0)
    total = np.sum(snr_e**2)
    return {
        "snr": float(snr), "period": float(period), "t0": float(t0), "dur_h": float(dur * 24),
        "depth_ppm": float(amp * 1e6), "transits": int(np.sum(cover >= MIN_COVERAGE)),
        "share": float(snr_e.max() ** 2 / total) if total > 0 else 1.0,
    }


def top_peaks(lc: dict, freq, nbin, stat, code, side: int, t_ref: float) -> list[dict]:
    snr = np.sqrt(stat[:, side].astype(np.float64))
    period = 1 / freq
    peaks = []
    for _ in range(TOP_PEAKS):
        m = int(np.argmax(snr))
        if snr[m] <= 0:
            break
        k, i = DUR_BINS[code[m, side] // NBMAX], code[m, side] % NBMAX
        p, nb = period[m], nbin[m]
        t0 = t_ref + (i + k / 2) / nb * p
        t0 += p * np.round((np.median(lc["t"]) - t0) / p)
        peaks.append(check_peak(lc, p, t0, k * p / nb, -1 if side == 0 else 1))
        snr[np.abs(period - p) < PEAK_SEPARATION * p] = 0
    return peaks


def passes(peak: dict) -> bool:
    return peak["transits"] >= MIN_TRANSITS and peak["share"] <= MAX_EVENT_SHARE and peak["snr"] > 0


def search_star(t, flux, sector, rad, mass, bls, p_min=P_MIN, p_max=P_MAX) -> tuple[dict, list[dict]]:
    if not (np.isfinite(rad) and rad > 0):
        rad = mass if np.isfinite(mass) and mass > 0 else 0.5
    if not (np.isfinite(mass) and mass > 0):
        mass = rad  # for K and M dwarfs, mass in solar units is close to radius
    lc = detrend(t, flux, sector)
    t_ref = lc["t"].min()
    baseline = lc["t"].max() - t_ref
    row = {"n_points": len(lc["t"]), "n_sectors": len(np.unique(sector)), "span_d": round(baseline, 1),
           "noise_ppm": round(lc["noise_ppm"], 1), "n_flare": lc["n_flare"], "n_freq": 0}
    if baseline < (MIN_TRANSITS - 1) * p_min:
        row.update(gate1="fail", reason="baseline too short")
        return row, []
    freq, nbin, klo, khi = grid(baseline, rad, mass, p_min, p_max)
    row["n_freq"] = len(freq)
    # Scale to O(1) for float32 on the GPU; the statistic doesn't change.
    s0 = float(np.median(1 / np.sqrt(lc["w"])))
    stat, code = bls(lc["t"] - t_ref, lc["y"] / s0, lc["w"] * s0**2, freq, nbin, klo, khi)

    peaks = []
    best = {}
    for side, name in ((0, "dip"), (1, "bump")):
        found = top_peaks(lc, freq, nbin, stat, code, side, t_ref)
        peaks += [{"side": name, "rank": n + 1, **p} for n, p in enumerate(found)]
        row[f"top_{name}_snr"] = round(found[0]["snr"], 2) if found else 0.0
        best[name] = next((p for p in found if passes(p)), None)
        for key in ("snr", "period", "t0", "dur_h", "depth_ppm", "transits", "share"):
            row[f"{name}_{key}"] = best[name][key] if best[name] else ""

    ok, reason = gate(row)
    row.update(gate1="pass" if ok else "fail", reason=reason)
    return row, peaks


def num(v) -> float | None:
    return None if v in ("", None) else float(v)


def gate(row: dict, floor: float = SNR_FLOOR, margin: float = MARGIN, flipped: bool = False) -> tuple[bool, str]:
    """Gate 1 on a results row; flipped=True runs the identical gate with dips and bumps swapped."""
    a, b = ("bump", "dip") if flipped else ("dip", "bump")
    snr = num(row.get(f"{a}_snr"))
    # The other side's best passing peak, else its raw top peak (the stricter choice).
    other = num(row.get(f"{b}_snr")) or num(row.get(f"top_{b}_snr")) or 0.0
    if snr is None:
        return False, f"no {a} with >= {MIN_TRANSITS} transits"
    if snr < floor:
        return False, f"{a} SNR {snr:.1f} < floor {floor:.1f}"
    if snr < margin * other:
        return False, f"{a} SNR {snr:.1f} < {margin} x flipped {other:.1f}"
    return True, f"{a} SNR {snr:.1f} >= floor {floor:.1f} and {margin} x flipped {other:.1f}"


def calibrate(results: Path, margin: float = MARGIN, rate: float = FALSE_PASS_RATE) -> dict:
    """Measure the SNR floor on a finished survey, re-apply gate 1 to every row, write gate1.json."""
    with results.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    searched = [r for r in rows if r["n_freq"] not in ("", "0")]
    for floor in np.arange(7.0, 100.0, 0.1):
        false = sum(gate(r, floor, margin, flipped=True)[0] for r in searched)
        if false <= rate * len(searched):
            break
    floor = round(float(floor), 1)
    for r in searched:
        ok, r["reason"] = gate(r, floor, margin)
        r["gate1"] = "pass" if ok else "fail"
    with results.open("w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    summary = {"stars": len(searched), "floor": floor, "margin": margin, "false_pass_rate": rate,
               "pass": sum(r["gate1"] == "pass" for r in searched), "flipped_pass": false}
    (results.parent / "gate1.json").write_text(json.dumps(summary, indent=1))
    log(f"gate 1 calibrated on {len(searched)} stars: floor {floor}, margin {margin}; "
        f"{summary['pass']} pass, {false} pass when flipped (expected false alarms)")
    return summary


# ---------------------------------------------------------------- self-test and driver

def synthetic(seed=1, period=3.217, depth=2500e-6, dur=0.08, noise=1000e-6, days=27.0):
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, 1 / 48) + 2000.0
    t = t[(t - t[0]) % 13.7 > 1.0]  # a mid-sector downlink gap
    flux = 1 + 0.004 * np.sin(2 * np.pi * t / 5.3) + rng.normal(0, noise, len(t))
    phase = (t - 2001.3 + period / 2) % period - period / 2
    flux[np.abs(phase) < dur / 2] -= depth
    return t, flux, np.full(len(t), 5, np.int16)


def selftest(bls) -> None:
    """GPU kernel against the NumPy reference, and recovery of an injected transit."""
    t, flux, sector = synthetic()
    lc = detrend(t, flux, sector)
    freq, nbin, klo, khi = grid(np.ptp(lc["t"]), 0.5, 0.5, 1.0, 8.0)
    args = (lc["t"] - lc["t"].min(), lc["y"] / 1e-3, lc["w"] * 1e-6, freq, nbin, klo, khi)
    s_ref, _ = bls_numpy(*args)
    s_gpu, _ = bls(*args)
    err = np.max(np.abs(s_gpu - s_ref) / np.maximum(s_ref, 1))
    row, _ = search_star(t, flux, sector, 0.5, 0.5, bls, 1.0, 8.0)
    log(f"selftest: {len(freq)} freqs, max rel diff GPU vs NumPy {err:.1e}; "
        f"injected P 3.217 -> {row['dip_period']}, gate1 {row['gate1']}")
    assert err < 1e-3, "GPU and NumPy BLS disagree"
    assert row["gate1"] == "pass" and abs(row["dip_period"] - 3.217) < 0.01, "injected transit not recovered"


RESULT_FIELDS = ["tic", "shard", "n_points", "n_sectors", "span_d", "noise_ppm", "n_flare", "n_freq",
                 "top_dip_snr", "top_bump_snr"] + \
    [f"{side}_{k}" for side in ("dip", "bump")
     for k in ("snr", "period", "t0", "dur_h", "depth_ppm", "transits", "share")] + \
    ["gate1", "reason", "seconds"]
PEAK_FIELDS = ["tic", "side", "rank", "snr", "period", "t0", "dur_h", "depth_ppm", "transits", "share"]


def fmt(v):
    return f"{v:.6g}" if isinstance(v, float) else v


def find_shards(root: str) -> dict[int, Path]:
    shards = {}
    for d, _, files in os.walk(root):  # one listing; input storage is slow per file
        for name in files:
            if name.startswith("lc_") and name.endswith(".npz") and ".tmp" not in name:
                shards[int(name[3:-4])] = Path(d) / name
    return dict(sorted(shards.items()))


def read_list(path: str | None) -> set[int]:
    return {int(x) for x in Path(path).read_text().split()} if path else set()


def run(args, part: int, nparts: int) -> None:
    stars = {}
    if args.stars:
        with open(args.stars, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter=" "):
                stars[int(r["tic"])] = (float(r["rad"] or "nan"), float(r["mass"] or "nan"))
    skip, only = read_list(args.skip), set(args.tics or [])
    shards = find_shards(args.lc)
    if args.shards:
        shards = {n: p for n, p in shards.items() if n in set(args.shards)}
    mine = [n for n in shards if n % nparts == part]

    if args.cpu:
        bls = bls_numpy
    else:
        try:
            bls = GpuBLS()
        except ImportError:
            log("CuPy not found: using the NumPy reference (slow)")
            bls = bls_numpy
    if bls is not bls_numpy:
        selftest(bls)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_part{part}" if nparts > 1 else ""
    res_path, peak_path = out / f"results{suffix}.csv", out / f"peaks{suffix}.csv"
    done = set()
    if res_path.exists():
        with res_path.open(newline="") as f:
            done = {int(r["tic"]) for r in csv.DictReader(f)}
    for path, fields in ((res_path, RESULT_FIELDS), (peak_path, PEAK_FIELDS)):
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.DictWriter(f, fields).writeheader()

    log(f"part {part + 1}/{nparts}: {len(mine)} shards, {len(done)} stars already done")
    n_done = 0
    for n in mine:
        data = np.load(shards[n])
        tics = [int(k[:-5]) for k in data.files if k.endswith("_time")]
        for tic in tics:
            if tic in done or tic in skip or (only and tic not in only):
                continue
            if (time.time() - T0) / 60 > args.max_minutes:
                log("time limit reached")
                return
            t1 = time.time()
            rad, mass = stars.get(tic, (np.nan, np.nan))
            row, peaks = search_star(data[f"{tic}_time"], data[f"{tic}_flux"], data[f"{tic}_sector"],
                                     rad, mass, bls)
            row.update(tic=tic, shard=n, seconds=round(time.time() - t1, 2))
            with res_path.open("a", newline="") as f:
                csv.DictWriter(f, RESULT_FIELDS).writerow({k: fmt(v) for k, v in row.items()})
            with peak_path.open("a", newline="") as f:
                w = csv.DictWriter(f, PEAK_FIELDS)
                for p in peaks:
                    w.writerow({k: fmt(v) for k, v in {"tic": tic, **p}.items()})
            n_done += 1
            if only or n_done % 25 == 0:
                log(f"shard {n} TIC {tic}: {row['n_freq']} freqs, {row['seconds']} s, "
                    f"dip {row['dip_snr'] and round(row['dip_snr'], 1)} @ {row['dip_period'] and round(row['dip_period'], 4)} d, "
                    f"gate1 {row['gate1']} ({row['reason']})  [{n_done} stars]")
        log(f"shard {n} done")


def merge(out: Path, nparts: int) -> None:
    for name in ("results", "peaks"):
        parts = [out / f"{name}_part{i}.csv" for i in range(nparts)]
        with (out / f"{name}.csv").open("w", newline="") as f:
            for i, p in enumerate(parts):
                lines = p.read_text().splitlines(keepends=True) if p.exists() else []
                f.writelines(lines if i == 0 else lines[1:])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lc", help="folder searched (recursively) for lc_NNN.npz shards")
    ap.add_argument("--out")
    ap.add_argument("--calibrate", help="only re-run gate 1 calibration on this results.csv")
    ap.add_argument("--stars", help="survey_stars.txt: tic rad mass")
    ap.add_argument("--skip", help="TIC IDs never to search (the held-back set)")
    ap.add_argument("--shards", type=int, nargs="*", help="only these shard numbers")
    ap.add_argument("--tics", type=int, nargs="*", help="only these stars")
    ap.add_argument("--gpus", type=int, default=1, help="one worker process per GPU")
    ap.add_argument("--part", help="internal: i/n, this worker's share of the shards")
    ap.add_argument("--cpu", action="store_true", help="NumPy reference instead of the GPU kernel")
    ap.add_argument("--max-minutes", type=float, default=660, help="stop cleanly after this long")
    args = ap.parse_args()
    if args.calibrate:
        calibrate(Path(args.calibrate))
        return
    if not (args.lc and args.out):
        ap.error("--lc and --out are required")

    if args.part:
        i, n = map(int, args.part.split("/"))
        run(args, i, n)
        return
    if args.gpus > 1:
        # Separate processes rather than multiprocessing: forking after CUDA init can hang.
        procs = [subprocess.Popen([sys.executable, "-u", __file__, *sys.argv[1:], "--part", f"{i}/{args.gpus}"],
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(i)))
                 for i in range(args.gpus)]
        codes = [p.wait() for p in procs]
        merge(Path(args.out), args.gpus)
        log(f"workers exited {codes}; merged")
        if any(codes):
            sys.exit(1)
    else:
        run(args, 0, 1)
    if not args.tics:
        calibrate(Path(args.out) / "results.csv")


if __name__ == "__main__":
    main()
