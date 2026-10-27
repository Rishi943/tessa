"""Code vetting for a flagged signal: the checks that need no model.

Each check returns plain numbers, and fixed rules turn them into a verdict. Clear eclipsing
binaries and known objects are closed here, so the agent (and its model calls) only sees
what survives. The rules are deliberately conservative: a real planet should never be
rejected by them, so anything borderline goes to the agent instead.

Run: python -m pipeline.vet TIC --results results.csv [--lc data/cache/lc] [--blind]
Prints the checks as JSON. --blind skips the known-object match, for scoring the vetting on
stars whose answer is already known.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from . import catalogs
from .paths import CACHE, DATA
from .search import central_duration, check_peak, detrend

R_EARTH_PER_RSUN = 109.08
HARMONICS = (1 / 3, 1 / 2, 1, 2, 3)
PERIOD_MATCH = 0.01  # relative tolerance for a period (or harmonic) match
NEIGHBOUR_ARCSEC = 150  # about 7 TESS pixels: a star this close can leak into the aperture

# Rejection rules
MAX_PLANET_REARTH = 25.0  # about 2.2 Jupiter radii after undoing dilution
SECONDARY_SNR = 7.0
SECONDARY_OVER_BUMP = 1.5
ODD_EVEN_SIGMA = 4.0
MAX_DURATION_RATIO = 3.0  # observed / central duration for this star


def load_lc(tic: int, lc_dir: Path) -> dict[str, np.ndarray]:
    manifest = lc_dir / "manifest.csv"
    if not manifest.exists():  # the survey's shards stay on Kaggle; their manifest is kept locally
        manifest = DATA / "fetch_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as f:
        shard = next(int(r["shard"]) for r in csv.DictReader(f) if int(r["tic"]) == tic)
    data = np.load(lc_dir / f"lc_{shard:03d}.npz")
    return {k: data[f"{tic}_{k}"] for k in ("time", "flux", "sector", "source")}


def star_info(tic: int) -> dict:
    """From the target table, or straight from the TIC for stars outside the survey pool."""
    cols = ["tic", "tmag", "teff", "rad", "mass", "distance", "contratio", "ruwe",
            "phot_variable_flag", "non_single_star", "ra", "dec", "upcoming_sectors"]
    rows = pd.read_parquet(DATA / "targets.parquet", columns=cols).query("tic == @tic")
    if len(rows):
        info = rows.iloc[0].to_dict()
        info["upcoming_sectors"] = [int(s) for s in info["upcoming_sectors"]]
    else:
        from astroquery.mast import Catalogs
        r = Catalogs.query_criteria(catalog="Tic", ID=tic)[0]
        info = {"tic": tic, "tmag": r["Tmag"], "teff": r["Teff"], "rad": r["rad"], "mass": r["mass"],
                "distance": r["d"], "contratio": r["contratio"], "ra": r["ra"], "dec": r["dec"]}
    info = {k: (v.item() if hasattr(v, "item") else v) for k, v in info.items()}
    info = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in info.items()}
    if info["mass"] is None and info["rad"]:
        info["mass"], info["mass_estimated"] = info["rad"], True  # dwarfs: mass close to radius
    return info


def phase_offset(t: np.ndarray, period: float, t0: float) -> np.ndarray:
    """Time from the nearest mid-transit, in days."""
    return (t - t0 + period / 2) % period - period / 2


def box_depth(lc: dict, mask: np.ndarray) -> tuple[float, float]:
    """Depth (relative flux, positive = dimmer) and its error inside a mask."""
    r = lc["w"][mask].sum()
    if r == 0:
        return np.nan, np.nan
    return float(-np.sum(lc["w"][mask] * lc["y"][mask]) / r), float(1 / np.sqrt(r))


def red_noise(lc: dict, sig: dict, mask: np.ndarray) -> float:
    """How far real errors exceed formal ones on the transit's timescale (1 = white noise).

    Cut the out-of-transit data into boxes one transit long and measure each box's depth in
    units of its formal error; white noise scatters those by 1, spots and systematics by more."""
    out = mask & (np.abs(phase_offset(lc["t"], sig["period"], sig["t0"])) > sig["dur"])
    if out.sum() < 20:
        return 1.0
    _, box = np.unique(np.floor(lc["t"][out] / sig["dur"]), return_inverse=True)
    r = np.bincount(box, lc["w"][out])
    snr = -np.bincount(box, lc["w"][out] * lc["y"][out]) / np.sqrt(r)
    snr = snr[np.bincount(box) >= 2]
    if len(snr) < 10:
        return 1.0
    return max(1.0, 1.4826 * float(np.median(np.abs(snr - np.median(snr)))))


def odd_even(lc: dict, sig: dict) -> dict:
    epoch = np.round((lc["t"] - sig["t0"]) / sig["period"]).astype(np.int64)
    inside = np.abs(phase_offset(lc["t"], sig["period"], sig["t0"])) < sig["dur"] / 2
    d_odd, e_odd = box_depth(lc, inside & (epoch % 2 == 1))
    d_even, e_even = box_depth(lc, inside & (epoch % 2 == 0))
    beta = red_noise(lc, sig, np.ones(len(lc["t"]), bool))
    return {"depth_odd_ppm": d_odd * 1e6, "depth_even_ppm": d_even * 1e6, "red_noise": beta,
            "diff_sigma": abs(d_odd - d_even) / np.hypot(e_odd, e_even) / beta}


def secondary(lc: dict, sig: dict) -> dict:
    """Best dimming (and, for scale, best brightening) away from the transit, same duration."""
    p, dur = sig["period"], sig["dur"]
    phases = np.arange(0.2, 0.8, dur / p / 4)
    found = [box_depth(lc, np.abs(phase_offset(lc["t"], p, sig["t0"] + ph * p)) < dur / 2) for ph in phases]
    depth = np.array([d for d, _ in found])
    snr = np.nan_to_num(np.array([d / e for d, e in found]))
    best = int(np.argmax(snr))
    d_half, e_half = box_depth(lc, np.abs(phase_offset(lc["t"], p, sig["t0"] + 0.5 * p)) < dur / 2)
    return {"snr": float(snr[best]), "phase": float(phases[best]), "depth_ppm": float(depth[best] * 1e6),
            "snr_at_half": d_half / e_half, "best_brightening_snr": float(-snr.min())}


def sector_depths(lc: dict, sig: dict) -> dict:
    inside = np.abs(phase_offset(lc["t"], sig["period"], sig["t0"])) < sig["dur"] / 2
    rows = []
    for s in np.unique(lc["sector"]):
        d, e = box_depth(lc, inside & (lc["sector"] == s))
        if np.isfinite(d):
            rows.append((int(s), d, e * red_noise(lc, sig, lc["sector"] == s)))
    d = np.array([r[1] for r in rows])
    e = np.array([r[2] for r in rows])
    mean = np.sum(d / e**2) / np.sum(1 / e**2)
    chi2 = float(np.sum(((d - mean) / e) ** 2))
    return {"per_sector": [{"sector": s, "depth_ppm": round(x * 1e6), "err_ppm": round(y * 1e6)} for s, x, y in rows],
            "chi2": chi2, "dof": len(rows) - 1,
            "p_constant": float(stats.chi2.sf(chi2, len(rows) - 1)) if len(rows) > 1 else 1.0}


def size(sig: dict, star: dict) -> dict:
    """Planet radius implied by the depth, after undoing dilution by nearby stars."""
    contratio = star["contratio"] or 0.0
    depth = sig["depth_ppm"] * 1e-6 * (1 + contratio)
    ratio = float(np.sqrt(max(depth, 0)))
    return {"undiluted_depth_ppm": depth * 1e6, "rp_over_rstar": ratio,
            "rp_rearth": ratio * star["rad"] * R_EARTH_PER_RSUN}


def duration(sig: dict, star: dict) -> dict:
    expected = float(central_duration(np.array(sig["period"]), star["rad"], star["mass"]))
    return {"observed_h": sig["dur"] * 24, "central_h": expected * 24, "ratio": sig["dur"] / expected}


def harmonic(p_cat: float, p_sig: float) -> float | None:
    if not np.isfinite(p_cat) or p_cat <= 0:
        return None
    r = p_cat / p_sig
    return next((h for h in HARMONICS if abs(r - h) / h < PERIOD_MATCH), None)


def known(tic: int, star: dict, sig: dict) -> list[dict]:
    """This star or a close neighbour in the TOI, CTOI, confirmed-planet or EB lists."""
    hits = []
    lists = [("TOI", catalogs.toi(), "toi"), ("CTOI", catalogs.ctoi(), "ctoi")]
    for name, df, key in lists:
        sep = 3600 * np.hypot((df["ra"] - star["ra"]) * np.cos(np.radians(star["dec"])), df["dec"] - star["dec"])
        for i, r in df[(df["tic"] == tic) | (sep < NEIGHBOUR_ARCSEC)].iterrows():
            hits.append({"list": name, "id": str(r[key]), "tic": int(r["tic"]), "period": float(r["period"]),
                         "separation_arcsec": round(float(sep[i]), 1), "harmonic": harmonic(r["period"], sig["period"])})
    for name, df, key in [("confirmed", catalogs.confirmed(), "planet"), ("TESS EB", catalogs.tess_ebs(), "tic")]:
        for _, r in df[df["tic"] == tic].iterrows():
            hits.append({"list": name, "id": str(r[key]), "tic": tic, "period": float(r["period"]),
                         "separation_arcsec": 0.0, "harmonic": harmonic(r["period"], sig["period"])})
    return hits


def verdict(checks: dict) -> tuple[str, list[str]]:
    """'reject' with the failed checks, or 'agent' when nothing clearly rules it out."""
    why = []
    for k in checks["known"]:
        if k["harmonic"] is not None:
            where = "this star" if k["tic"] == checks["tic"] else f"TIC {k['tic']}, {k['separation_arcsec']:.0f}\" away"
            why.append(f"known: {k['list']} {k['id']} ({where}) at {k['harmonic']:g}x this period")
    s = checks["secondary"]
    if s["snr"] >= SECONDARY_SNR and s["snr"] >= SECONDARY_OVER_BUMP * s["best_brightening_snr"]:
        why.append(f"secondary eclipse: SNR {s['snr']:.1f} at phase {s['phase']:.2f}")
    if checks["odd_even"]["diff_sigma"] >= ODD_EVEN_SIGMA:
        why.append(f"odd and even transits differ by {checks['odd_even']['diff_sigma']:.1f} sigma")
    if checks["size"]["rp_rearth"] > MAX_PLANET_REARTH:
        why.append(f"too large for a planet: {checks['size']['rp_rearth']:.0f} Earth radii")
    if checks["duration"]["ratio"] > MAX_DURATION_RATIO:
        why.append(f"transit {checks['duration']['ratio']:.1f}x longer than this star allows")
    return ("reject" if why else "agent"), why


def vet(tic: int, sig: dict, lc_dir: Path, blind: bool = False) -> dict:
    """sig: period, t0, dur (days), depth_ppm, snr, from the search."""
    raw = load_lc(tic, lc_dir)
    lc = detrend(raw["time"], raw["flux"], raw["sector"])
    star = star_info(tic)
    checks = {
        "tic": tic, "star": star, "signal": {**sig, "dur_h": sig["dur"] * 24},
        "refit": check_peak(lc, sig["period"], sig["t0"], sig["dur"], -1),
        "odd_even": odd_even(lc, sig), "secondary": secondary(lc, sig),
        "sectors": sector_depths(lc, sig), "size": size(sig, star),
        "duration": duration(sig, star), "known": [] if blind else known(tic, star, sig),
    }
    checks["code_verdict"], checks["reasons"] = verdict(checks)
    return checks


def signal_from_results(tic: int, results: Path) -> dict:
    with results.open(newline="") as f:
        r = next(r for r in csv.DictReader(f) if int(r["tic"]) == tic)
    return {"period": float(r["dip_period"]), "t0": float(r["dip_t0"]), "dur": float(r["dip_dur_h"]) / 24,
            "depth_ppm": float(r["dip_depth_ppm"]), "snr": float(r["dip_snr"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tic", type=int)
    ap.add_argument("--results", required=True, help="search results.csv with this star's signal")
    ap.add_argument("--lc", default=str(CACHE / "lc"), help="folder with the lc_NNN.npz shards")
    ap.add_argument("--blind", action="store_true", help="skip the known-object match")
    args = ap.parse_args()
    out = vet(args.tic, signal_from_results(args.tic, Path(args.results)), Path(args.lc), args.blind)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
