"""System 1: a calibrated P(planet) from the numeric checks, with the checks that drove each score.

XGBoost on the same split as the fine-tune: trees chosen by grouped CV on train (by host star), a sigmoid
calibration fitted on val, test scored once. Trained on confirmed planets vs false positives (APC left out:
their label is the question itself); APC are scored and reported, not trained on. Code-rejected TOIs are
in training too (they carry the rules' misses), and results are reported on the agent-bound subset,
which is where System 1 sits in production.

Run: python -m finetune.system1 [--name NAME] [--multis] [--no-star] [--search N]
     (outputs per version in data/finetune/system1/*-NAME.json; compare versions with finetune/compare.py)
     python -m finetune.system1 --laya data/finetune/kaggle_laya/laya_preds.json   (score the Laya challenger)
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import beta

from pipeline.paths import DATA

FOLDER = DATA / "finetune"
OUT = FOLDER / "system1"
STAR_SHORTCUTS = ("tmag", "distance")


def ratio(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def features(c: dict) -> dict:
    """The checks as numbers. None stays missing (XGBoost learns where missing goes)."""
    st, r, oe, s = c["star"], c["refit"], c["odd_even"], c["secondary"]
    size, dur, pc, sh = c["size"], c["duration"], c["phase_curve"], c["shape"]
    nb, gaps, sec = c["neighbours"], c["gaps"], c["sectors"]
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


def other_tois() -> dict[int, int]:
    """TOIs per host star in the ExoFOP snapshot, any disposition: multiplicity, the best-known planet prior.
    Uses no outcome, but in production it needs our search to look for more signals on the star."""
    counts: dict[int, int] = {}
    with open(FOLDER / "toi_exofop_2026-10-28.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            counts[int(row["TIC ID"])] = counts.get(int(row["TIC ID"]), 0) + 1
    return counts


CLUE_GROUPS = ("centroid", "fit", "gaia", "eb")


def log10(v):
    return math.log10(v) if v is not None and v > 0 else None


def clue_features(d: dict, clue: dict, gaia: dict, ebs: dict, groups) -> dict:
    """The clue job's extras (finetune/clues.py) as numbers, by group."""
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
        from finetune.clues import eb_match
        period = c["refit"]["period"]
        target = str(c["star"].get("source_id"))
        near = {str(n.get("source_id")) for n in (c["neighbours"].get("closest_that_could") or [])}
        near |= {str(n.get("source_id")) for n in ((c.get("centroid") or clue.get("centroid") or {}).get("suspects") or [])}
        on = eb_match(period, ebs.get(target))
        off = [m for m in (eb_match(period, ebs.get(s)) for s in near - {target}) if m is not None]
        x.update(gaia_eb_target_match=on, gaia_eb_neighbour_match=min(off) if off else None)
    return x


def load(multis: bool = False, groups=()):
    counts = other_tois() if multis else {}
    clues, gaia, ebs = {}, {}, {}
    if groups:
        clues = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (FOLDER / "clues").glob("*.json")}
        for p in FOLDER.glob("gaia_clues_*.json"):
            g = json.loads(p.read_text(encoding="utf-8"))
            gaia.update(g["extra"])
            ebs.update(g["eb"])
    rows = []
    for path in sorted((FOLDER / "vet").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        x = features(d["checks"])
        if multis:
            x["other_tois_on_star"] = counts.get(int(d["tic"]), 1) - 1
        if groups:
            x.update(clue_features(d, clues.get(d["toi"], {}), gaia, ebs, groups))
        rows.append({"toi": d["toi"], "tic": d["tic"], "label": d["label"], "code": d["checks"]["code_verdict"], "x": x})
    return rows


def matrix(rows, names):
    return np.array([[np.nan if r["x"][n] is None else float(r["x"][n]) for n in names] for r in rows])


BASE = dict(objective="binary:logistic", eval_metric="logloss", random_state=3407, n_jobs=4)
PARAMS = dict(BASE, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
              reg_lambda=1.0)
ALPHAS = (0.01, 0.02, 0.05)


def conformal_cut(planet_scores, alpha: float) -> float:
    """Split conformal: reject below the k-th lowest calibration planet score, k = floor((n + 1) alpha).
    A new planet from the same population then falls below it with probability k / (n + 1) <= alpha."""
    s = np.sort(planet_scores)
    k = int(np.floor((len(s) + 1) * alpha))
    return float(s[k - 1]) if k >= 1 else 0.0


def certified(n: int, alpha: float, confidence: float = 0.95) -> float:
    """The planet-loss rate the same cut keeps to with `confidence` (not just on average): the realised
    loss of the k-th order statistic is Beta(k, n + 1 - k)."""
    k = int(np.floor((n + 1) * alpha))
    return float(beta.ppf(confidence, k, n + 1 - k)) if k >= 1 else 0.0


def report(name, y, p, cuts, code):
    """Ranking (AUC), calibration (Brier), and what rejecting below each conformal cut costs and catches:
    with the code rules' rejections standing (today's pipeline) and System 1 alone (rules as inputs only)."""
    out = {"n": int(len(y)), "planets": int(y.sum()), "auc": round(roc_auc_score(y, p), 4),
           "brier": round(brier_score_loss(y, p), 4),
           "rules: planets lost": round(float(code[y == 1].mean()), 3),
           "rules: fps caught": round(float(code[y == 0].mean()), 3)}
    for alpha, cut in cuts.items():
        for how, drop in (("rules+S1", code | (p < cut)), ("S1 alone", p < cut)):
            out[f"{how} {alpha:.0%}: planets lost"] = round(float(drop[y == 1].mean()), 3)
            out[f"{how} {alpha:.0%}: fps caught"] = round(float(drop[y == 0].mean()), 3)
    print(f"\n{name}: " + ", ".join(f"{k} {v}" for k, v in out.items()))
    return out


def binary(rs):
    return [r for r in rs if r["label"]["target"] in ("gate2", "reject")]


def y(rs):
    return np.array([r["label"]["target"] == "gate2" for r in rs], int)


def scorecard(raw, split):
    """Platt scaling and reject cuts fitted on val, then the test subsets. `raw(rows)` gives margins."""
    val, test = binary(split["val"]), binary(split["test"])
    cal = LogisticRegression(C=1e6).fit(raw(val).reshape(-1, 1), y(val))
    prob = lambda rs: cal.predict_proba(raw(rs).reshape(-1, 1))[:, 1]  # noqa: E731
    planets = prob([r for r in val if r["label"]["target"] == "gate2"])
    cuts = {a: conformal_cut(planets, a) for a in ALPHAS}
    results = {"cuts": cuts, "certified_95": {a: round(certified(len(planets), a), 4) for a in ALPHAS}}
    print("conformal cuts on", len(planets), "val planets:",
          ", ".join(f"{a:.0%} -> P < {c:.3f} (95%-certified loss <= {results['certified_95'][a]:.1%})" for a, c in cuts.items()))
    for name, rs in (("val", val), ("test", test), ("test, agent-bound", [r for r in test if r["code"] == "agent"]),
                     ("test, K/M hosts", [r for r in test if (r["label"].get("teff") or 9999) < 5300]),
                     ("test, alerted 2024+", [r for r in test if r["label"].get("recent")])):
        results[name] = report(name, y(rs), prob(rs), cuts, np.array([r["code"] == "reject" for r in rs]))
    apc = [r for r in split["test"] if r["label"]["target"] == "flag"]
    results["test APC median P(planet)"] = round(float(np.median(prob(apc))), 3)
    print(f"test APC ({len(apc)}): median P(planet) {results['test APC median P(planet)']}")
    return cal, results


def score_laya(path: Path, split) -> None:
    """Laya's P(planet) from the Kaggle job, scored like XGBoost (margins = logit of its probability)."""
    preds = json.loads(path.read_text(encoding="utf-8"))
    for name, p in preds.items():
        print(f"\n===== Laya {name}")
        logit = lambda rs: np.array([np.log(q / (1 - q)) for q in (np.clip(p[r["toi"]], 1e-6, 1 - 1e-6) for r in rs)])  # noqa: E731
        _, results = scorecard(logit, split)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"results-laya-{name}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


def grouped_cv(params, X, Y, folds, names):
    """Trees and log loss by early stopping on grouped CV."""
    cv = xgb.cv(params, xgb.DMatrix(X, Y, feature_names=names), num_boost_round=3000, early_stopping_rounds=50,
                folds=folds, seed=3407)
    loss = list(cv["test-logloss-mean"])
    return len(loss), loss[-1]


def search(X, Y, folds, names, trials: int):
    """Random search over the usual XGBoost knobs, scored by grouped-CV log loss on train."""
    rng = np.random.default_rng(3407)
    best = (PARAMS, *grouped_cv(PARAMS, X, Y, folds, names))
    print(f"default: log loss {best[2]:.4f} at {best[1]} trees")
    for i in range(trials):
        p = dict(BASE, max_depth=int(rng.integers(2, 8)), learning_rate=float(10 ** rng.uniform(-2.3, -1)),
                 subsample=float(rng.uniform(0.5, 1)), colsample_bytree=float(rng.uniform(0.3, 1)),
                 min_child_weight=float(10 ** rng.uniform(0, 1.3)), reg_lambda=float(10 ** rng.uniform(-1, 1.5)),
                 reg_alpha=float(10 ** rng.uniform(-3, 0.5)), gamma=float(rng.uniform(0, 2)))
        rounds, loss = grouped_cv(p, X, Y, folds, names)
        if loss < best[2]:
            best = (p, rounds, loss)
            print(f"trial {i}: log loss {loss:.4f} at {rounds} trees "
                  + json.dumps({k: round(v, 3) for k, v in p.items() if k not in BASE}))
    return best[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-star", action="store_true", help="drop brightness and distance (shortcut check)")
    ap.add_argument("--laya", type=Path, help="score Laya's predictions (laya_preds.json) instead of training")
    ap.add_argument("--search", type=int, default=0, help="random-search N parameter sets (saved to params.json)")
    ap.add_argument("--multis", action="store_true", help="add multiplicity: other TOIs on the same star")
    ap.add_argument("--clues", default="", help=f"comma-separated clue groups from finetune/clues.py: {','.join(CLUE_GROUPS)} or all")
    ap.add_argument("--name", help="version name for the outputs (default: base, no-star or multis)")
    args = ap.parse_args()
    groups = CLUE_GROUPS if args.clues == "all" else tuple(g for g in args.clues.split(",") if g)
    rows = load(args.multis, groups)
    split = {s: [r for r in rows if r["label"]["split"] == s] for s in ("train", "val", "test")}
    if args.laya:
        return score_laya(args.laya, split)
    names = [n for n in rows[0]["x"] if not (args.no_star and n in STAR_SHORTCUTS)]
    train = binary(split["train"])
    name = args.name or ("no-star" if args.no_star else "multis" if args.multis else "base")
    OUT.mkdir(parents=True, exist_ok=True)

    X, Y, groups = matrix(train, names), y(train), [r["tic"] for r in train]
    folds = list(GroupKFold(5).split(X, Y, groups))
    params_path = OUT / "params.json"  # tuned once on the base features; every version reuses them
    if args.search:
        params_path.write_text(json.dumps(search(X, Y, folds, names, args.search), indent=1), encoding="utf-8")
    params = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else PARAMS
    rounds, loss = grouped_cv(params, X, Y, folds, names)
    print(f"train {len(train)} (planets {Y.sum()}), CV log loss {loss:.4f} at {rounds} trees")
    model = xgb.train(params, xgb.DMatrix(X, Y, feature_names=names), num_boost_round=rounds)

    # out-of-fold margins for train: every labelled TOI gets a score from a model that never saw its star
    oof = np.zeros(len(train))
    for fit, held in folds:
        m = xgb.train(params, xgb.DMatrix(X[fit], Y[fit], feature_names=names), num_boost_round=rounds)
        oof[held] = m.predict(xgb.DMatrix(X[held], feature_names=names), output_margin=True)
    print(f"train out-of-fold AUC {roc_auc_score(Y, oof):.4f}")

    raw = lambda rs: model.predict(xgb.DMatrix(matrix(rs, names), feature_names=names), output_margin=True)  # noqa: E731
    cal, results = scorecard(raw, split)
    results.update({"features": names, "trees": rounds, "params": params, "name": name, "clues": list(groups)})
    # CV+-style cut: val planets plus out-of-fold train planets as the calibration set (860 instead of 110)
    calib = cal.predict_proba(np.concatenate([oof[Y == 1], raw([r for r in binary(split["val"]) if r["label"]["target"] == "gate2"])]).reshape(-1, 1))[:, 1]
    test = binary(split["test"])
    pt, yt = cal.predict_proba(raw(test).reshape(-1, 1))[:, 1], y(test)
    results["cv_plus"] = {}
    for a in ALPHAS:
        c = conformal_cut(calib, a)
        results["cv_plus"][a] = {"cut": round(c, 4), "certified_95": round(certified(len(calib), a), 4),
                                 "test planets lost": round(float((pt[yt == 1] < c).mean()), 3),
                                 "test fps caught": round(float((pt[yt == 0] < c).mean()), 3)}
    print(f"CV+ cuts on {len(calib)} planets:", json.dumps(results["cv_plus"]))
    scored = split["val"] + split["test"]
    margins = np.concatenate([oof, raw(scored)])
    preds = dict(zip((r["toi"] for r in train + scored), cal.predict_proba(margins.reshape(-1, 1))[:, 1].round(4).tolist()))

    gain = model.get_score(importance_type="gain")
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:12]
    print("top checks by gain:", ", ".join(f"{k} {v:.1f}" for k, v in top))
    results["gain"] = {k: round(v, 2) for k, v in top}

    model.save_model(OUT / f"xgb-{name}.json")
    (OUT / f"calibration-{name}.json").write_text(json.dumps({"a": float(cal.coef_[0][0]), "b": float(cal.intercept_[0])}))
    (OUT / f"results-{name}.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    (OUT / f"preds-{name}.json").write_text(json.dumps(preds), encoding="utf-8")  # P(planet): train out-of-fold, val, test


if __name__ == "__main__":
    main()
