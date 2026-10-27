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
import contextlib
import csv
import json
import sys
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
PHASE_BINS = 10
VARIABILITY_RATIO = 0.5  # out-of-transit swing at this period, as a fraction of the dip depth,
VARIABILITY_CHI2 = 100  # at this significance (red-noise-scaled chi2 over PHASE_BINS - 1 dof)

# Shape and neighbours (reported, never rejected on: the agent weighs them)
INGRESS_FRACTIONS = np.round(np.arange(0.05, 0.51, 0.05), 2)  # 0.5 is a V
FLAT_MAX = 0.25  # fits with ingress fraction up to this count as flat-bottomed
MAX_ECLIPSE_DEPTH = 0.5  # deepest eclipse a neighbouring binary is assumed to make
PLAUSIBLE_ECLIPSE = 0.25  # eclipses this shallow are common, so such a neighbour is a real worry
TESS_PIXEL_ARCSEC = 21.0
GAIA_EPOCH, TIC_EPOCH = 2016.0, 2000.0


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
    cols = ["tic", "source_id", "tmag", "teff", "rad", "mass", "distance", "contratio", "ruwe",
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


def phase_curve(lc: dict, sig: dict) -> dict:
    """Does the star brighten and dim at this period outside the dip?

    A planet adds at most a few ppm out of transit. A spotted star rotating at this period, or a
    contact or ellipsoidal binary, swings by as much as the dip itself, and the box search then
    picks the minimum of that swing as a "transit"."""
    p = sig["period"]
    x = phase_offset(lc["t"], p, sig["t0"]) / p
    out = np.abs(x) * p > sig["dur"]
    b = np.minimum(((x[out] + 0.5) * PHASE_BINS).astype(int), PHASE_BINS - 1)
    sw = np.bincount(b, lc["w"][out], PHASE_BINS)
    filled = sw > 0
    m = np.bincount(b, lc["w"][out] * lc["y"][out], PHASE_BINS)[filled] / sw[filled]
    mean = np.sum(sw[filled] * m) / sw[filled].sum()
    chi2 = float(np.sum(sw[filled] * (m - mean) ** 2)) / red_noise(lc, sig, np.ones(len(lc["t"]), bool)) ** 2
    amp = float(m.max() - m.min()) * 1e6
    return {"amplitude_ppm": amp, "ratio_to_depth": amp / sig["depth_ppm"], "chi2": chi2, "dof": int(filled.sum()) - 1}


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


def trapezoid(x: np.ndarray, total: float, ingress: float) -> np.ndarray:
    """1 at full depth, 0 out of transit; ingress is a fraction of the total duration."""
    tau = ingress * total
    return np.clip((total / 2 - np.abs(x)) / tau, 0, 1)


def shape(lc: dict, sig: dict) -> dict:
    """Flat-bottomed (planet-like) or V-shaped (grazing, often a binary)?

    Trapezoids are fitted to the folded transit over a grid of total durations, ingress
    fractions and small mid-time shifts, with depth and baseline solved linearly. v_sigma
    compares the best V with the best flat-bottomed fit, in red-noise-scaled sigma: positive
    favours the V. At low SNR, or with 30-minute cadence on a short transit, it stays near 0."""
    dur = sig["dur"]
    x = phase_offset(lc["t"], sig["period"], sig["t0"])
    near = np.abs(x) < 1.5 * dur
    x, y, w = x[near], lc["y"][near], lc["w"][near]
    s1, sy = w.sum(), np.sum(w * y)
    gain = {}
    for total in dur * np.linspace(0.6, 1.8, 13):
        for ingress in INGRESS_FRACTIONS:
            for shift in dur * np.linspace(-0.2, 0.2, 5):
                m = trapezoid(x - shift, total, ingress)
                sm, smm, smy = np.sum(w * m), np.sum(w * m * m), np.sum(w * m * y)
                cov, var = smy - sm * sy / s1, smm - sm * sm / s1
                if var > 0 and cov < 0:  # a dip
                    g = cov * cov / var
                    if g > gain.get(ingress, (0,))[0]:
                        gain[ingress] = (g, total)
    if not gain:
        return {"best_ingress_fraction": None, "v_sigma": 0.0, "t14_h": None}
    best = max(gain, key=lambda k: gain[k][0])
    flat = max((gain[k][0] for k in gain if k <= FLAT_MAX), default=0.0)
    v = gain.get(0.5, (0.0,))[0]
    beta = red_noise(lc, sig, np.ones(len(lc["t"]), bool))
    return {"best_ingress_fraction": float(best), "t14_h": float(gain[best][1] * 24),
            "v_sigma": float(np.sign(v - flat) * np.sqrt(abs(v - flat)) / beta)}


def gaia_field(tic: int, star: dict) -> tuple[pd.DataFrame, pd.Series] | None:
    """Gaia DR3 stars within NEIGHBOUR_ARCSEC (cached), with "mag" in the band closest to TESS and
    "sep" from the target at the TIC's epoch, and the target's own row; None if it isn't found."""
    path = CACHE / "gaia_neighbours" / f"{tic}.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        with contextlib.redirect_stdout(sys.stderr):  # the module prints a release banner on import
            from astroquery.gaia import Gaia
        query = ("SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag AS g, phot_rp_mean_mag AS rp "
                 "FROM gaiadr3.gaia_source WHERE 1 = CONTAINS(POINT(ra, dec), "
                 f"CIRCLE({star['ra']}, {star['dec']}, {NEIGHBOUR_ARCSEC / 3600}))")
        df = Gaia.launch_job(query, verbose=False).get_results().to_pandas()
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
    # TIC positions are at epoch 2000, Gaia DR3 at 2016: move Gaia stars back before comparing.
    if "pmra" not in df:
        df["pmra"] = df["pmdec"] = np.nan
    years = GAIA_EPOCH - TIC_EPOCH
    dec = df["dec"] - df["pmdec"].fillna(0) * years / 3.6e6
    ra = df["ra"] - df["pmra"].fillna(0) * years / 3.6e6 / np.cos(np.radians(df["dec"]))
    df["sep"] = 3600 * np.hypot((ra - star["ra"]) * np.cos(np.radians(star["dec"])), dec - star["dec"])
    df["mag"] = df["rp"].fillna(df["g"])  # RP is closest to the TESS band
    df = df.dropna(subset=["mag"]).sort_values("sep")
    mine = df[df["source_id"] == star.get("source_id")]
    close = mine if len(mine) else df[df["sep"] < 5]
    if close.empty:
        return None
    return df, close.loc[close["mag"].idxmin()]


def neighbours(tic: int, star: dict, sig: dict) -> dict:
    """Gaia DR3 stars near the target bright enough to make this dip if they were binaries.

    A neighbour could be the source if the eclipse it would need, the observed depth scaled
    up by the brightness ratio, is at most MAX_ECLIPSE_DEPTH. Only centroid or difference
    imaging can then tell; this just says whether such a star exists and how close it is."""
    field = gaia_field(tic, star)
    if field is None:
        return {"target_found": False}
    df, me = field
    others = df.drop(index=me.name)
    need = sig["depth_ppm"] * 1e-6 * 10 ** (0.4 * (others["mag"] - me["mag"]))
    able = others[need <= MAX_ECLIPSE_DEPTH].assign(need=need[need <= MAX_ECLIPSE_DEPTH])
    return {"target_found": True, "gaia_stars_within": len(others), "radius_arcsec": NEIGHBOUR_ARCSEC,
            "could_cause": int(len(able)),
            "could_cause_within_1px": int((able["sep"] < TESS_PIXEL_ARCSEC).sum()),
            "plausible_within_1px": int(((able["sep"] < TESS_PIXEL_ARCSEC) & (able["need"] <= PLAUSIBLE_ECLIPSE)).sum()),
            "closest_that_could": [{"source_id": int(r.source_id), "sep_arcsec": round(float(r.sep), 1),
                                    "delta_mag": round(float(r.mag - me["mag"]), 2), "eclipse_needed": round(float(r.need), 4)}
                                   for r in able.head(3).itertuples()]}


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
    pc = checks["phase_curve"]
    if pc["ratio_to_depth"] >= VARIABILITY_RATIO and pc["chi2"] >= VARIABILITY_CHI2:
        why.append(f"the star varies by {pc['amplitude_ppm']:.0f} ppm at this period outside the dip "
                   f"({pc['ratio_to_depth']:.1f}x the dip): rotation or a contact binary")
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
        "duration": duration(sig, star), "phase_curve": phase_curve(lc, sig), "shape": shape(lc, sig),
        "neighbours": neighbours(tic, star, sig),
        "known": [] if blind else known(tic, star, sig),
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
