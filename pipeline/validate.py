"""The tests every candidate's preprint reports, beyond gate 2's checks.

1. Transit fit with bootstrap errors (pipeline/fit.py): radius ratio, impact parameter, a/R*, and the star
   density the transit shape implies against the catalogue star's (a blend or a binary rarely matches).
   Errors come from refitting transits resampled with replacement.
2. Gaia DR3 binarity: RUWE, radial-velocity scatter, image doubling, astrometric excess noise, and Gaia's
   own radius (FLAME) against the TIC's.
3. Leave-one-season-out: each observing season is hidden, the same blind search runs on the rest, and the
   period it finds predicts the hidden season's transits; the dip must be there.
4. The upside-down ceiling: the best "dip" the same search finds in this star's flipped light curve.
TRICERATOPS runs separately (pipeline/fpp.py, Kaggle).

Run: python -m pipeline.validate TIC [--boot 100]    writes data/validation/TIC.json
     Off the laptop: python -m pipeline.validate TIC [TIC ...] --export DIR, then on Kaggle (CPU) the pipeline
     package with those files beside it: python -m pipeline.validate --job; then --import DIR with its output.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from .fit import G_CGS, gaia_extra, transit_fit
from .paths import DATA

OUT = DATA / "validation"
SEASON_GAP_D = 60  # sectors closer than this belong to one season
R_EARTH_PER_R_SUN = 109.08


def exact_source_id(tic: int) -> int | None:
    """Gaia DR3 source_id from the target table: the vet files' copies went through float and lost digits."""
    import pandas as pd
    t = pd.read_parquet(DATA / "targets.parquet", columns=["tic", "source_id"])
    row = t[t["tic"] == tic]
    return int(row["source_id"].iloc[0]) if len(row) else None


def bootstrap_fit(lc: dict, sig: dict, star: dict, n: int, seed: int = 20261028) -> dict:
    best = transit_fit(lc, sig, star)
    if best.get("status") != "ok":
        return {"best": best}
    epoch = np.round((lc["t"] - sig["t0"]) / sig["period"])
    near = np.abs(lc["t"] - sig["t0"] - epoch * sig["period"]) < 2 * sig["dur"]
    epochs = np.unique(epoch[near])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        pick = rng.choice(epochs, len(epochs), replace=True)
        idx = np.concatenate([np.flatnonzero(near & (epoch == e)) for e in pick])
        sub = {k: lc[k][idx] for k in ("t", "y", "w", "span")}
        f = transit_fit(sub, sig, star)
        if f.get("status") == "ok":
            draws.append(f)

    def spread(key):
        v = np.array([d[key] for d in draws if d.get(key) is not None])
        return {"median": float(np.median(v)), "lo": float(np.percentile(v, 16)), "hi": float(np.percentile(v, 84))} if len(v) else None

    out = {"best": best, "n_boot": len(draws)}
    for key in ("k", "b", "ars", "rho_fit", "rho_ratio", "star_density_penalty", "t14_h"):
        out[key] = spread(key)
    if out["k"] and star.get("rad"):
        out["rp_rearth"] = {q: v * star["rad"] * R_EARTH_PER_R_SUN for q, v in out["k"].items()}
    return out


def seasons(sector: np.ndarray, t: np.ndarray) -> list[list[int]]:
    mids = sorted((float(np.median(t[sector == s])), int(s)) for s in np.unique(sector))
    groups = [[mids[0][1]]]
    for (a, _), (b, s) in zip(mids, mids[1:]):
        if b - a > SEASON_GAP_D:
            groups.append([])
        groups[-1].append(s)
    return groups


def leave_one_season_out(raw: dict, star: dict, sig: dict) -> list[dict]:
    from .search import bls_numpy, check_peak, detrend, search_star
    out = []
    for hidden in seasons(raw["sector"], raw["time"]):
        rest = ~np.isin(raw["sector"], hidden)
        row, _ = search_star(raw["time"][rest], raw["flux"][rest], raw["sector"][rest], star["rad"], star["mass"], bls_numpy)
        r = {"hidden_sectors": hidden, "blind_period": row.get("dip_period") or None, "blind_snr": row.get("dip_snr") or None,
             "gate1": row.get("gate1")}
        p = r["blind_period"]
        r["same_period"] = bool(p and abs(p / sig["period"] - 1) < 2e-3)
        if r["same_period"]:
            m = np.isin(raw["sector"], hidden)
            lc_h = detrend(raw["time"][m], raw["flux"][m], raw["sector"][m])
            # the blind ephemeris predicts the hidden season; measure the dip at exactly those times
            peak = check_peak(lc_h, float(p), float(row["dip_t0"]), float(row["dip_dur_h"]) / 24, -1)
            r.update(predicted_depth_ppm=round(peak["depth_ppm"]), predicted_snr=round(peak["snr"], 1),
                     predicted_transits=peak["transits"])
        out.append(r)
    return out


def inputs(tic: int) -> dict:
    """Everything the tests need for one candidate, from the local survey files: the light curve, the star,
    the refitted ephemeris, the exact Gaia ID and gate 1's scores. As JSON it is a Kaggle job's input."""
    from .fpp import candidate
    from .tools import light_curve
    cand, _ = candidate(tic)
    checks, raw, _ = light_curve(tic)
    eph = cand["when"]["ephemeris"]
    site = DATA / "site" / "stars" / f"{tic}.json"
    g1 = json.loads(site.read_text(encoding="utf-8")).get("gate1", {}) if site.exists() else {}
    return {"tic": tic, "star": dict(checks["star"]), "source_id": exact_source_id(tic),
            "sig": {"period": eph["period"], "t0": eph["t0_btjd"], "dur": eph["t14_h"] / 24, "depth_ppm": cand["signal"]["depth_ppm"]},
            "flipped": {"dip_snr": g1.get("dip_snr"), "flipped_snr": g1.get("flipped_snr")},
            "raw": {k: np.asarray(raw[k]).tolist() for k in ("time", "flux", "sector")}}


def run(d: dict, n_boot: int) -> dict:
    """The tests on one candidate's inputs (inputs()); needs only numpy, scipy and astroquery's Gaia."""
    from .search import detrend
    raw = {k: np.asarray(v, dtype=int if k == "sector" else float) for k, v in d["raw"].items()}
    lc = detrend(raw["time"], raw["flux"], raw["sector"])
    star, sig, sid = d["star"], d["sig"], d["source_id"]
    out = {"tic": d["tic"]}
    out["fit"] = bootstrap_fit(lc, sig, star, n_boot)
    g = gaia_extra([sid]).get(str(sid), {}) if sid else {}
    out["gaia"] = {"source_id": str(sid) if sid else None, "ruwe": star.get("ruwe"), "non_single_star": star.get("non_single_star"),
                   **{k: g.get(k) for k in ("rv_chisq_pvalue", "rv_renormalised_gof", "rv_amplitude_robust", "rv_nb_transits",
                                            "radial_velocity_error", "ipd_frac_multi_peak", "ipd_gof_harmonic_amplitude",
                                            "astrometric_excess_noise_sig", "radius_flame", "mass_flame", "radius_gspphot")},
                   "tic_radius": star.get("rad")}
    out["seasons"] = leave_one_season_out(raw, star, sig)
    out["flipped"] = d["flipped"]
    return out


def validate(tic: int, n_boot: int) -> dict:
    out = run(inputs(tic), n_boot)
    save(out)
    return out


def save(out: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{out['tic']}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tic", type=int, nargs="*")
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--export", type=Path, help="write each candidate's inputs (validate_input_TIC.json) to this folder")
    ap.add_argument("--job", action="store_true", help="run every validate_input_*.json in the working folder (Kaggle)")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"), help="where --job writes validation_TIC.json")
    ap.add_argument("--import", dest="results", type=Path, help="save a job's validation_*.json into data/validation")
    args = ap.parse_args()
    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        for tic in args.tic:
            (args.export / f"validate_input_{tic}.json").write_text(json.dumps(inputs(tic), default=float), encoding="utf-8")
            print(f"exported TIC {tic}")
    elif args.job:
        args.out.mkdir(parents=True, exist_ok=True)
        for path in sorted(Path(".").glob("validate_input_*.json")):
            d = json.loads(path.read_text(encoding="utf-8"))
            t0 = time.time()
            try:
                v = run(d, args.boot)
            except Exception as e:  # one candidate's failure must not stop the others
                v = {"tic": d["tic"], "error": f"{e.__class__.__name__}: {e}"}
            (args.out / f"validation_{d['tic']}.json").write_text(json.dumps(v, indent=1, default=float), encoding="utf-8")
            print(f"{time.strftime('%H:%M:%S')} TIC {d['tic']}: {v.get('error', 'done')} in {(time.time() - t0) / 60:.1f} min", flush=True)
    elif args.results:
        for path in sorted(args.results.glob("validation_*.json")):
            v = json.loads(path.read_text(encoding="utf-8"))
            if "error" in v:
                print(f"TIC {v['tic']}: {v['error']}")
                continue
            save(v)
            report(v)
    else:
        for tic in args.tic:
            report(validate(tic, args.boot))


def report(v: dict) -> None:
    print(f"TIC {v['tic']}")
    f = v["fit"]
    if f.get("rho_ratio"):
        print(f"fit: k {f['k']['median']:.4f} [{f['k']['lo']:.4f}, {f['k']['hi']:.4f}], b {f['b']['median']:.2f}, "
              f"density ratio {f['rho_ratio']['median']:.2f} [{f['rho_ratio']['lo']:.2f}, {f['rho_ratio']['hi']:.2f}], "
              f"star-density penalty {f['best'].get('star_density_penalty', float('nan')):.1f} (n_boot {f['n_boot']})")
    print("gaia:", {k: v["gaia"][k] for k in ("ruwe", "rv_chisq_pvalue", "rv_renormalised_gof", "ipd_frac_multi_peak", "radius_flame", "tic_radius")})
    for s in v["seasons"]:
        print("season", s)
    print("flipped:", v["flipped"])


if __name__ == "__main__":
    main()
