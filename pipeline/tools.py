"""Tools the agent can call to settle a flag, one per kind of doubt. Every result is measured by
code and saved to data/evidence/TIC.json, where gate 2 reads it: the agent chooses what to look
at, but only the numbers decide.

Run: python -m pipeline.tools NAME TIC      (prints JSON; NAME is one of TOOLS)
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from .paths import CACHE, DATA
from .search import central_duration, detrend
from .vet import R_EARTH_PER_RSUN, box_depth, load_lc, phase_offset, red_noise

# Where each batch's vetting and light curves live
BATCHES = {"survey": (DATA / "vet" / "survey", CACHE / "lc"), "batch2": (DATA / "vet" / "batch2", CACHE / "lc_batch2"),
           "labelled": (DATA / "vet" / "labelled", CACHE / "lc_labelled")}
SOURCE_NAMES = {0: "SPOC 2-min", 1: "TESS-SPOC", 2: "QLP"}
EVIDENCE = DATA / "evidence"
P_CONSTANT = 0.01  # same line as the sector check
TRANSIT_OUTLIER_SIGMA = 4.0
MAX_ROTATION_D = 15.0
ROTATION_BIN_D = 0.5 / 24
ROTATION_FAP = 1e-3  # a sector's best periodogram peak counts below this false-alarm probability


def locate(tic: int) -> tuple[dict, Path]:
    """The star's code checks and its light-curve folder."""
    for vet_dir, lc_dir in BATCHES.values():
        path = vet_dir / f"{tic}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), lc_dir
    raise LookupError(f"TIC {tic} has not been vetted")


def light_curve(tic: int) -> tuple[dict, dict, dict]:
    checks, lc_dir = locate(tic)
    raw = load_lc(tic, lc_dir)
    return checks, raw, detrend(raw["time"], raw["flux"], raw["sector"])


def centroid_all(tic: int) -> dict:
    """Difference imaging on every sector instead of the latest six: for shallow dips and blends
    the first pass called inconclusive."""
    from .centroid import where
    checks, _ = locate(tic)
    c = where(tic, checks["signal"], max_sectors=99)
    return {k: c.get(k) for k in ("verdict", "why", "offset_arcsec", "err_arcsec", "offset_sigma", "sectors_used",
                                  "on_star", "suspects")}


def sector_depths_by_source(tic: int) -> dict:
    """Depth per sector with the light-curve product it came from. QLP and TESS-SPOC correct for
    neighbouring stars' light differently, so a depth that is steady within each product but
    differs between them is a calibration difference, not a sign of a binary."""
    checks, raw, lc = light_curve(tic)
    sig = checks["signal"]
    inside = np.abs(phase_offset(lc["t"], sig["period"], sig["t0"])) < sig["dur"] / 2
    source_of = {int(s): int(np.bincount(raw["source"][raw["sector"] == s].astype(int)).argmax())
                 for s in np.unique(raw["sector"])}
    rows = []
    for s in np.unique(lc["sector"]):
        d, e = box_depth(lc, inside & (lc["sector"] == s))
        if np.isfinite(d):
            rows.append({"sector": int(s), "source": SOURCE_NAMES[source_of[int(s)]], "depth_ppm": round(d * 1e6),
                         "err_ppm": round(e * red_noise(lc, sig, lc["sector"] == s) * 1e6)})

    def steady(group: list[dict]) -> tuple[float, float, float]:
        d = np.array([r["depth_ppm"] for r in group], float)
        e = np.array([r["err_ppm"] for r in group], float)
        mean = np.sum(d / e**2) / np.sum(1 / e**2)
        chi2 = float(np.sum(((d - mean) / e) ** 2))
        p = float(stats.chi2.sf(chi2, len(d) - 1)) if len(d) > 1 else 1.0
        return float(mean), float(1 / np.sqrt(np.sum(1 / e**2))), p

    by_source = []
    for name in sorted({r["source"] for r in rows}):
        mean, err, p = steady([r for r in rows if r["source"] == name])
        by_source.append({"source": name, "sectors": sum(r["source"] == name for r in rows),
                          "depth_ppm": round(mean), "err_ppm": round(err), "p_constant": p})
    _, _, p_all = steady(rows)
    within = all(g["p_constant"] >= P_CONSTANT for g in by_source if g["sectors"] > 1)
    tested = any(g["sectors"] > 1 for g in by_source)
    if p_all >= P_CONSTANT:
        verdict = "steady"
    elif within and tested and len(by_source) > 1:
        verdict = "differs_by_source"
    else:
        verdict = "changes_within_a_source"
    return {"verdict": verdict, "p_constant_all": p_all, "by_source": by_source, "per_sector": rows}


def transit_depths(tic: int) -> dict:
    """Depth of every single transit. A planet's transits scatter around one depth; a signal made
    by a few deep events (flares' aftermath, systematics, a one-off eclipse) does not."""
    checks, _, lc = light_curve(tic)
    sig = checks["signal"]
    epoch = np.round((lc["t"] - sig["t0"]) / sig["period"]).astype(np.int64)
    inside = np.abs(phase_offset(lc["t"], sig["period"], sig["t0"])) < sig["dur"] / 2
    beta = red_noise(lc, sig, np.ones(len(lc["t"]), bool))
    events = []
    for n in np.unique(epoch[inside]):
        d, e = box_depth(lc, inside & (epoch == n))
        if np.isfinite(d):
            events.append((int(n), d, e * beta))
    d = np.array([x[1] for x in events])
    e = np.array([x[2] for x in events])
    mean = np.sum(d / e**2) / np.sum(1 / e**2)
    z = (d - mean) / e
    snr2 = np.clip(d / e, 0, None) ** 2
    top3 = float(np.sort(snr2)[-3:].sum() / snr2.sum()) if snr2.sum() > 0 else 1.0
    outliers = [{"epoch": events[i][0], "btjd": round(sig["t0"] + events[i][0] * sig["period"], 3),
                 "depth_ppm": round(d[i] * 1e6), "sigma": round(float(z[i]), 1)}
                for i in np.argsort(-np.abs(z)) if abs(z[i]) >= TRANSIT_OUTLIER_SIGMA][:5]
    chi2 = float(np.sum(z**2))
    return {"transits": len(events), "mean_depth_ppm": round(mean * 1e6), "chi2": chi2, "dof": len(events) - 1,
            "p_consistent": float(stats.chi2.sf(chi2, len(events) - 1)) if len(events) > 1 else 1.0,
            "share_of_top_3": round(top3, 3), "outliers": outliers}


def stellar_radius(tic: int) -> dict:
    """Gaia DR3's own radius and mass for the star (FLAME, from its luminosity and temperature),
    against the TIC's: the expected duration and the planet's size depend on them."""
    checks, _ = locate(tic)
    star, sig = checks["star"], checks["signal"]
    out = {"tic_radius": star["rad"], "tic_mass": star["mass"]}
    if star.get("source_id"):
        with contextlib.redirect_stdout(sys.stderr):
            from astroquery.gaia import Gaia
            r = Gaia.launch_job("SELECT radius_flame, mass_flame, radius_gspphot, teff_gspphot "
                                f"FROM gaiadr3.astrophysical_parameters WHERE source_id = {int(star['source_id'])}",
                                verbose=False).get_results()
        if len(r):
            out.update({k: (None if np.ma.is_masked(r[k][0]) else float(r[k][0])) for k in r.colnames})
    rows = []
    for name, rad, mass in [("TIC", star["rad"], star["mass"]),
                            ("Gaia FLAME", out.get("radius_flame"), out.get("mass_flame") or star["mass"]),
                            ("Gaia GSP-Phot", out.get("radius_gspphot"), star["mass"])]:
        if rad:
            expected = float(central_duration(np.array(sig["period"]), rad, mass))
            depth = sig["depth_ppm"] * 1e-6 * (1 + (star["contratio"] or 0))
            rows.append({"radius_from": name, "rad": round(rad, 3), "mass": round(mass, 3),
                         "duration_ratio": round(sig["dur"] / expected, 2),
                         "rp_rearth": round(float(np.sqrt(depth)) * rad * R_EARTH_PER_RSUN, 2)})
    return {**out, "options": rows}


def rotation(tic: int) -> dict:
    """The star's own brightness cycle (spots rotating into view), from a periodogram of the
    light curve before detrending, sector by sector. A dip at the rotation period or a simple
    multiple of it may be the spots, not a transit."""
    from astropy.timeseries import LombScargle
    checks, raw, _ = light_curve(tic)
    p_sig = checks["signal"]["period"]
    found = []
    for s in np.unique(raw["sector"]):
        m = (raw["sector"] == s) & np.isfinite(raw["flux"])
        if m.sum() < 100:
            continue
        # half-hour bins: spots change over days, and binning lifts them above the photon noise
        k = np.floor((raw["time"][m] - raw["time"][m].min()) / ROTATION_BIN_D).astype(int)
        n = np.bincount(k)
        t = (np.bincount(k, raw["time"][m]) / np.maximum(n, 1))[n > 0]
        y = (np.bincount(k, raw["flux"][m]) / np.maximum(n, 1))[n > 0]
        ls = LombScargle(t, y)
        freq = np.linspace(1 / MAX_ROTATION_D, 1 / 0.1, 20000)
        power = ls.power(freq)
        best = int(np.argmax(power))
        found.append({"sector": int(s), "period_d": round(float(1 / freq[best]), 3),
                      "false_alarm": float(ls.false_alarm_probability(power[best], minimum_frequency=freq[0],
                                                                       maximum_frequency=freq[-1])),
                      "amplitude_ppm": round(float(np.percentile(y, 95) - np.percentile(y, 5)) * 1e6)})
    strong = [f for f in found if f["false_alarm"] < ROTATION_FAP]
    periods = np.array([f["period_d"] for f in strong])
    consensus = float(np.median(periods)) if len(periods) else None
    ratio = p_sig / consensus if consensus else None
    multiple = next((h for h in (1 / 3, 1 / 2, 1, 2, 3) if ratio and abs(ratio - h) / h < 0.03), None)
    return {"rotation_d": consensus, "signal_period_d": p_sig, "signal_over_rotation": ratio,
            "signal_at_rotation_multiple": multiple, "sectors": found}


def predict(tic: int) -> dict:
    """When the next transits can be checked: TESS sectors to come and nights on the ground."""
    from .when import when
    checks, lc_dir = locate(tic)
    w = when(tic, checks["signal"], lc_dir)
    return {k: w.get(k) for k in ("testable", "timing_err_now_min", "next_tess", "ground_nights")} | \
        {"next_ground": (w.get("ground") or [None])[0], "period": w["ephemeris"].get("period"),
         "period_err": w["ephemeris"].get("period_err")}


def known(tic: int) -> dict:
    """Is this signal already known? Fresh TOI, CTOI, planet and binary lists, VSX, Gaia, SIMBAD."""
    from .novelty import novelty
    checks, _ = locate(tic)
    return novelty(tic, checks["signal"])


TOOLS = {f.__name__: f for f in (centroid_all, sector_depths_by_source, transit_depths, stellar_radius, rotation,
                                 predict, known)}


def run(name: str, tic: int) -> dict:
    result = TOOLS[name](tic)
    path = EVIDENCE / f"{tic}.json"
    EVIDENCE.mkdir(exist_ok=True)
    evidence = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    evidence[name] = result
    path.write_text(json.dumps(evidence, indent=1, default=float), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("name", choices=list(TOOLS))
    ap.add_argument("tic", type=int)
    args = ap.parse_args()
    print(json.dumps(run(args.name, args.tic), default=float))


if __name__ == "__main__":
    main()
