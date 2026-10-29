"""System 1 in the pipeline: a calibrated P(planet) from the code checks, and the checks that drove it.

The XGBoost model from finetune/system1.py (all clue groups), trained on TESS candidates whose fate
follow-up settled. Its reject cut is split conformal on 860 calibration planets (out-of-fold train + val):
below it, a planet like those is lost 2% of the time (at most 2.8%, with 95% confidence). Odd/even,
secondary eclipse and size are inputs here, not vetoes; the junk filters in vet.verdict stay hard.

The raw P assumes the training mix (about half planets), not the survey's base rate, so the case file
shows it as a rank: more planet-like than X% of the calibration planets.

Run: python -m pipeline.system1 VET_DIR [VET_DIR ...] --clues DIR [--out CSV]
rescores vet files made before System 1, with clues from the backfill job (pipeline/clues.py).
"""
import argparse
import contextlib
import csv
import json
import math
import sys
from functools import cache
from pathlib import Path

import numpy as np

from .paths import DATA

MODEL = DATA / "finetune" / "system1"
NAME = "clues"
ALPHA = "0.02"
CLUE_GROUPS = ("centroid", "fit", "gaia", "eb")
DRIVERS = 4

LABELS = {
    "tmag": "TESS magnitude", "teff": "star temperature (K)", "star_radius": "star radius (Sun = 1)",
    "star_mass": "star mass (Sun = 1)", "distance": "distance (pc)", "contamination": "light from neighbours (TIC)",
    "ruwe": "Gaia RUWE", "gaia_variable": "Gaia variable flag", "non_single_star": "Gaia non-single-star flag",
    "snr": "dip SNR", "period": "period (d)", "duration_h": "duration (h)", "depth_ppm": "depth (ppm)",
    "transits": "transits", "odd_even_sigma": "odd/even difference (sigma)",
    "odd_even_fraction": "odd/even depth difference (fraction)", "red_noise": "red noise",
    "secondary_snr": "secondary eclipse SNR", "secondary_over_depth": "secondary depth / dip depth",
    "secondary_phase_offset": "secondary's distance from phase 0.5", "secondary_over_bump": "secondary SNR / brightening SNR",
    "sector_p_constant": "p(same depth in every sector)", "sector_chi2_dof": "depth scatter between sectors",
    "sectors": "sectors", "planet_radius": "planet radius (Earth = 1)", "radius_ratio": "planet / star radius",
    "duration_ratio": "duration / expected", "variability_over_depth": "out-of-dip swing / depth",
    "variability_chi2_dof": "out-of-dip swing (chi2/dof)", "ingress_fraction": "ingress fraction (0.5 = V)",
    "v_shape_sigma": "V-shape (sigma)", "gaia_neighbours": "Gaia stars within 150\"",
    "neighbours_could_cause": "neighbours that could fake it", "could_cause_within_1px": "such neighbours within a pixel",
    "plausible_within_1px": "plausible blends within a pixel", "closest_could_sep": "closest such neighbour (\")",
    "closest_could_dmag": "its magnitude difference", "near_gap_share": "share of transits next to data gaps",
    "gap_p_chance": "p(that many next to gaps)", "centroid_offset_sigma": "centroid offset (sigma)",
    "centroid_offset_arcsec": "centroid offset (\")", "centroid_off_target": "centroid off target",
    "fit_b": "fitted impact parameter", "fit_k": "fitted planet / star radius",
    "fit_log_density_ratio": "log(fitted / catalogue star density)", "fit_chi2_red": "transit fit chi2/dof",
    "star_density_penalty": "star-density penalty (chi2)", "star_density_penalty_share": "star-density penalty share",
    "rv_log_p": "log p(Gaia RV constant)", "rv_gof": "Gaia RV scatter (GOF)", "rv_amplitude": "Gaia RV amplitude (km/s)",
    "rv_error": "Gaia RV error (km/s)", "ipd_multi_peak": "Gaia double-image share (%)",
    "ipd_harmonic": "Gaia image asymmetry", "excess_noise_sig": "Gaia astrometric excess noise (sig)",
    "radius_tic_over_flame": "TIC / Gaia FLAME radius", "gaia_eb_target_match": "Gaia EB period match (target)",
    "gaia_eb_neighbour_match": "Gaia EB period match (neighbour)",
}


def ratio(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def log10(v):
    return math.log10(v) if v is not None and v > 0 else None


def features(c: dict) -> dict:
    """The checks as numbers. None stays missing (XGBoost learns where missing goes)."""
    st, r, oe, s = c["star"], c["refit"], c["odd_even"], c["secondary"]
    size, dur, pc, sh = c["size"], c["duration"], c["phase_curve"], c["shape"]
    nb, gaps, sec = c["neighbours"], c.get("gaps") or {}, c["sectors"]  # vet files from before the gaps check lack it
    cen = c.get("centroid") or {}
    a, b = oe["depth_odd_ppm"], oe["depth_even_ppm"]
    closest = (nb.get("closest_that_could") or [{}])[0]
    return {
        "tmag": st.get("tmag"), "teff": st.get("teff"), "star_radius": st.get("rad"), "star_mass": st.get("mass"),
        "distance": st.get("distance"), "contamination": st.get("contratio"), "ruwe": st.get("ruwe"),
        "gaia_variable": {"VARIABLE": 1, "NOT_AVAILABLE": 0}.get(st.get("phot_variable_flag")),
        "non_single_star": st.get("non_single_star"),
        "snr": r["snr"], "period": r["period"], "duration_h": r["dur_h"], "depth_ppm": r["depth_ppm"],
        "transits": r["transits"],
        "odd_even_sigma": oe["diff_sigma"],
        "odd_even_fraction": abs(a - b) / max(abs(a), abs(b)) if a is not None and b is not None and max(abs(a), abs(b)) else None,
        "red_noise": oe["red_noise"],
        "secondary_snr": s["snr"], "secondary_over_depth": ratio(s["depth_ppm"], r["depth_ppm"]),
        "secondary_phase_offset": abs(s["phase"] - 0.5) if s.get("phase") is not None else None,
        "secondary_over_bump": ratio(s["snr"], s["best_brightening_snr"]),
        "sector_p_constant": sec.get("p_constant"), "sector_chi2_dof": ratio(sec.get("chi2"), sec.get("dof")),
        "sectors": len(sec.get("per_sector") or []),
        "planet_radius": size["rp_rearth"], "radius_ratio": size["rp_over_rstar"],
        "duration_ratio": dur["ratio"],
        "variability_over_depth": pc["ratio_to_depth"], "variability_chi2_dof": ratio(pc["chi2"], pc["dof"]),
        "ingress_fraction": sh["best_ingress_fraction"], "v_shape_sigma": sh["v_sigma"],
        "gaia_neighbours": nb.get("gaia_stars_within"), "neighbours_could_cause": nb.get("could_cause"),
        "could_cause_within_1px": nb.get("could_cause_within_1px"), "plausible_within_1px": nb.get("plausible_within_1px"),
        "closest_could_sep": closest.get("sep_arcsec"), "closest_could_dmag": closest.get("delta_mag"),
        "near_gap_share": ratio(gaps.get("near_gap"), gaps.get("transits")), "gap_p_chance": gaps.get("p_chance"),
        "centroid_offset_sigma": cen.get("offset_sigma"), "centroid_offset_arcsec": cen.get("offset_arcsec"),
        "centroid_off_target": {"off_target": 1, "on_target": 0}.get(cen.get("verdict")),
    }


def clue_features(d: dict, clue: dict, gaia: dict, ebs: dict, groups) -> dict:
    """The clues beyond the code checks (transit fit, Gaia, the centroid the rules skipped) as numbers, by group.
    gaia and ebs are keyed by Gaia source_id as a string."""
    from .fit import eb_match
    c, x = d["checks"], {}
    if "centroid" in groups and not c.get("centroid") and clue.get("centroid"):
        cen = clue["centroid"]  # the centroid the rules skipped: the same three features as before
        x.update(centroid_offset_sigma=cen.get("offset_sigma"), centroid_offset_arcsec=cen.get("offset_arcsec"),
                 centroid_off_target={"off_target": 1, "on_target": 0}.get(cen.get("verdict")))
    if "fit" in groups:
        f = clue.get("fit") or {}
        ok = f.get("status") == "ok"
        dchi = f.get("delta_chi2") if ok else None
        pen = f.get("star_density_penalty") if ok else None
        x.update(fit_b=f.get("b") if ok else None, fit_k=f.get("k") if ok else None,
                 fit_log_density_ratio=log10(f.get("rho_ratio")) if ok else None,
                 fit_chi2_red=f.get("chi2_red") if ok else None,
                 star_density_penalty=pen, star_density_penalty_share=pen / dchi if pen is not None and dchi else None)
    if "gaia" in groups:
        g = gaia.get(str(c["star"].get("source_id"))) or {}
        x.update(rv_log_p=log10(g.get("rv_chisq_pvalue")), rv_gof=g.get("rv_renormalised_gof"),
                 rv_amplitude=g.get("rv_amplitude_robust"), rv_error=g.get("radial_velocity_error"),
                 ipd_multi_peak=g.get("ipd_frac_multi_peak"), ipd_harmonic=g.get("ipd_gof_harmonic_amplitude"),
                 excess_noise_sig=g.get("astrometric_excess_noise_sig"),
                 radius_tic_over_flame=c["star"]["rad"] / g["radius_flame"] if g.get("radius_flame") and c["star"].get("rad") else None)
    if "eb" in groups:
        period = c["refit"]["period"]
        target = str(c["star"].get("source_id"))
        near = {str(s) for s in suspects(c, clue)}
        on = eb_match(period, ebs.get(target))
        off = [m for m in (eb_match(period, ebs.get(s)) for s in near - {target}) if m is not None]
        x.update(gaia_eb_target_match=on, gaia_eb_neighbour_match=min(off) if off else None)
    return x


def suspects(c: dict, clue: dict | None = None) -> set[int]:
    """Gaia stars near the target that could make the dip: the neighbour check's and the centroid's."""
    cen = c.get("centroid") or (clue or {}).get("centroid") or {}
    found = {n.get("source_id") for n in c["neighbours"].get("closest_that_could") or []}
    found |= {n.get("source_id") for n in cen.get("suspects") or []}
    return {int(s) for s in found if s}


def clues(lc: dict, sig: dict, star: dict, checks: dict) -> dict:
    """What System 1 reads beyond the code checks, for one signal: the transit fit (star-density test),
    Gaia's binarity signs for the target, and Gaia's eclipsing-binary periods for it and its suspects."""
    from .fit import gaia_ebs, gaia_extra, transit_fit
    out = {"fit": transit_fit(lc, sig, star), "gaia": {}, "eb": {}}
    sid = star.get("source_id")
    if sid:
        with contextlib.redirect_stdout(sys.stderr):  # astroquery's release banner and the lookups' progress lines
            out["gaia"] = gaia_extra([int(sid)])
            out["eb"] = gaia_ebs(sorted(suspects(checks) | {int(sid)}))
    return out


@cache
def model() -> dict:
    """The booster, its calibration and cut, and the calibration planets' P (for the rank)."""
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(MODEL / f"xgb-{NAME}.json")
    cal = json.loads((MODEL / f"calibration-{NAME}.json").read_text(encoding="utf-8"))
    results = json.loads((MODEL / f"results-{NAME}.json").read_text(encoding="utf-8"))
    preds = json.loads((MODEL / f"preds-{NAME}.json").read_text(encoding="utf-8"))
    labels = json.loads((DATA / "finetune" / "labels.json").read_text(encoding="utf-8"))
    planets = np.sort([preds[r["toi"]] for r in labels
                       if r["target"] == "gate2" and r["split"] in ("train", "val") and r["toi"] in preds])
    return {"booster": booster, "names": booster.feature_names, "a": cal["a"], "b": cal["b"],
            "cut": results["cv_plus"][ALPHA]["cut"], "certified": results["cv_plus"][ALPHA]["certified_95"],
            "planets": planets}


def score(checks: dict, clue: dict | None = None) -> dict:
    """P(planet), whether it clears the cut, its rank among planets, and the checks that moved it most.
    clue defaults to the vet file's own (checks["clues"], from vet with System 1 on)."""
    import xgboost as xgb
    m = model()
    clue = clue if clue is not None else checks.get("clues") or {}
    x = features(checks)
    x.update(clue_features({"checks": checks}, clue, clue.get("gaia") or {}, clue.get("eb") or {}, CLUE_GROUPS))
    row = np.array([[np.nan if x.get(n) is None else float(x[n]) for n in m["names"]]])
    dm = xgb.DMatrix(row, feature_names=m["names"])
    margin = float(m["booster"].predict(dm, output_margin=True)[0])
    p = 1 / (1 + math.exp(-(m["a"] * margin + m["b"])))
    contrib = m["booster"].predict(dm, pred_contribs=True)[0][:-1]  # per check, in log-odds; the last is the bias
    top = np.argsort(-np.abs(contrib))[:DRIVERS]
    return {"p_planet": round(p, 4), "cut": m["cut"], "passes": p >= m["cut"],
            "planet_rank": round(float(np.searchsorted(m["planets"], p) / len(m["planets"])), 3),
            "drivers": [{"check": m["names"][i], "label": LABELS.get(m["names"][i], m["names"][i]),
                         "value": None if np.isnan(row[0, i]) else round(float(row[0, i]), 3),
                         "log_odds": round(float(contrib[i]), 2)} for i in top],
            "missing": int(np.isnan(row).sum()), "model": f"xgb-{NAME}"}


SIMILAR_K = 5


def ranks(train_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Each column's value as its rank among the training values, 0..1; NaN -> 0.5."""
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        ref = np.sort(train_x[:, j][np.isfinite(train_x[:, j])])
        col = x[:, j]
        out[:, j] = np.where(np.isfinite(col), np.searchsorted(ref, col) / max(len(ref), 1), 0.5)
    return out


def nearest(R_train: np.ndarray, weights: np.ndarray, v: np.ndarray, exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of the SIMILAR_K training rows closest to v (ranked features, gain-weighted distance)."""
    d = np.sqrt(((R_train - v) ** 2 * weights).sum(axis=1))
    if exclude is not None:
        d[exclude] = np.inf
    return np.argsort(d)[:SIMILAR_K], d


@cache
def similar_index() -> dict:
    """The training TOIs' features and outcomes (finetune/similar.py writes them)."""
    z = np.load(MODEL / f"similar-{NAME}.npz")
    meta = json.loads((MODEL / f"similar-{NAME}.json").read_text(encoding="utf-8"))
    return {"R": ranks(z["x"], z["x"]), "x": z["x"], "weights": z["weights"], "tics": z["tics"], "rows": meta}


def similar(checks: dict, clue: dict | None = None) -> list[dict]:
    """The 5 labelled TESS candidates whose checks look most like this signal's, and what follow-up found."""
    m, s = model(), similar_index()
    clue = clue if clue is not None else checks.get("clues") or {}
    x = features(checks)
    x.update(clue_features({"checks": checks}, clue, clue.get("gaia") or {}, clue.get("eb") or {}, CLUE_GROUPS))
    v = ranks(s["x"], np.array([[np.nan if x.get(n) is None else float(x[n]) for n in m["names"]]]))[0]
    idx, d = nearest(s["R"], s["weights"], v, s["tics"] == checks.get("tic"))
    return [{**s["rows"][i], "distance": round(float(d[i]), 4)} for i in idx]


def reason(s1: dict) -> str:
    """One line for a System 1 rejection: the score, the cut, and what pulled it down."""
    against = [d for d in s1["drivers"] if d["log_odds"] < 0][:3]
    said = ", ".join(d["label"] + " " + ("missing" if d["value"] is None else format(d["value"], "g")) for d in against)
    return (f"System 1: P(planet) {s1['p_planet']:.3f}, below the {s1['cut']:.3f} cut "
            f"(less planet-like than {1 - s1['planet_rank']:.0%} of planets)" + (f"; against it: {said}" if said else ""))


# --- rescoring vet files made before System 1 -------------------------------------------------------

FIELDS = ["tic", "batch", "status_before", "verdict_before", "verdict_now", "p_planet", "planet_rank", "change",
          "reasons_now", "drivers"]


def rescore(vet_dirs: list[Path], clue_dir: Path, write: bool = False) -> list[dict]:
    """Each vet file with the backfilled clues (centroid and gaps where it lacked them, fit, Gaia), scored and
    put through today's rules. write=True saves the updated checks over the vet files."""
    from .vet import finite, verdict
    g = json.loads((clue_dir / "gaia.json").read_text(encoding="utf-8"))
    rows = []
    for vet_dir in vet_dirs:
        status = {}
        if (vet_dir / "status.csv").exists():
            with (vet_dir / "status.csv").open(newline="", encoding="utf-8") as f:
                status = {int(r["tic"]): r["status"] for r in csv.DictReader(f)}
        for path in sorted(vet_dir.glob("[0-9]*.json")):
            c = json.loads(path.read_text(encoding="utf-8"))
            clue_path = clue_dir / "clues" / f"{c['tic']}.json"
            clue = json.loads(clue_path.read_text(encoding="utf-8")) if clue_path.exists() else {"error": "no clue file"}
            before = c["code_verdict"]
            for k in ("gaps", "centroid"):  # the checks the vet file lacked, measured by the backfill job
                if not c.get(k) and clue.get(k):
                    c[k] = clue[k]
            c["clues"] = {"fit": clue.get("fit") or {}, "gaia": g["extra"], "eb": g["eb"]}
            c["system1"] = score(c)
            c["similar"] = similar(c)
            c["clues"].update(gaia={k: v for k, v in g["extra"].items() if k == str(c["star"].get("source_id"))},
                              eb={k: v for k, v in g["eb"].items() if k in {str(s) for s in suspects(c) | {c["star"].get("source_id")}}})
            c["code_verdict"], c["reasons"] = verdict(c)
            s1 = c["system1"]
            rows.append({"tic": c["tic"], "batch": vet_dir.name, "status_before": status.get(c["tic"], ""),
                         "verdict_before": before, "verdict_now": c["code_verdict"], "p_planet": s1["p_planet"],
                         "planet_rank": s1["planet_rank"], "change": f"{before} -> {c['code_verdict']}" if before != c["code_verdict"] else "",
                         "reasons_now": "; ".join(c["reasons"]),
                         "drivers": "; ".join(f"{d['label']} {d['value']} ({d['log_odds']:+.2f})" for d in s1["drivers"]),
                         "clue_error": clue.get("error", "")})
            if write:
                path.write_text(json.dumps(finite(c), indent=1, default=float), encoding="utf-8")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vet_dirs", nargs="+", type=Path)
    ap.add_argument("--clues", type=Path, required=True, help="the backfill job's output (clues/ and gaia.json)")
    ap.add_argument("--out", type=Path, default=DATA / "system1" / "rescore.csv")
    ap.add_argument("--write", action="store_true", help="save the rescored checks over the vet files")
    args = ap.parse_args()
    rows = rescore(args.vet_dirs, args.clues, args.write)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS + ["clue_error"])
        w.writeheader()
        w.writerows(rows)
    moves = {}
    for r in rows:
        key = f"{r['verdict_before']} -> {r['verdict_now']}"
        moves[key] = moves.get(key, 0) + 1
    print(f"{len(rows)} signals ({sum(bool(r['clue_error']) for r in rows)} with a clue error): {moves}")
    for s in ("candidate", "known", "flagged", "rejected"):
        sub = [r for r in rows if r["status_before"] == s]
        if sub:
            print(f"  gate 2 {s} ({len(sub)}): {sum(r['verdict_now'] == 'reject' for r in sub)} now rejected by code, "
                  f"median P(planet) {np.median([r['p_planet'] for r in sub]):.3f}")
    print("candidates:")
    for r in rows:
        if r["status_before"] in ("candidate", "known"):
            print(f"  TIC {r['tic']} ({r['status_before']}): P(planet) {r['p_planet']:.3f}, more planet-like than "
                  f"{r['planet_rank']:.0%} of planets, {r['verdict_now']}; {r['drivers']}")
    print(f"wrote {args.out}" + ("; vet files updated" if args.write else ""))


if __name__ == "__main__":
    main()
