"""Export the survey for the site.

sky.json has every star in the pool, column by column, with a status code; summary.json has the
funnel and the counts; stars/TIC.json has, for every star that passed gate 1, its light curve
(binned and folded), the code checks, the agent's dossier, the gate 2 status and, for candidates,
the novelty check and the predictions.

Run: python -m pipeline.export [--out data/site] [--model muse-spark-1.3-contributor] [--prompt v5]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CACHE, DATA, ROOT
from .search import detrend
from .vet import load_lc, phase_offset

# (name, search results, vetting folder, light-curve shards) for each searched batch
RUNS = [("survey", DATA / "survey_results.csv", DATA / "vet" / "survey", CACHE / "lc"),
        ("batch2", DATA / "batch2_results.csv", DATA / "vet" / "batch2", CACHE / "lc_batch2")]
STATUS = {"pool": 0, "held back": 1, "searched": 2, "rejected": 3, "flagged": 4, "awaiting agent": 7,
          "candidate": 5, "known": 6}
BIN_D = 0.5 / 24  # light-curve bins for the page
FOLD_HALF = 3.0  # folded view: this many transit durations each side
FOLD_BINS = 60


def binned(t: np.ndarray, y: np.ndarray, w: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, ...]:
    i = np.digitize(t, edges) - 1
    ok = (i >= 0) & (i < len(edges) - 1)
    sw = np.bincount(i[ok], w[ok], len(edges) - 1)
    swy = np.bincount(i[ok], w[ok] * y[ok], len(edges) - 1)
    swt = np.bincount(i[ok], w[ok] * t[ok], len(edges) - 1)
    has = sw > 0
    return swt[has] / sw[has], swy[has] / sw[has], 1 / np.sqrt(sw[has])


def light_curve(lc: dict, sig: dict) -> dict:
    """The whole light curve in BIN_D bins (per sector, so gaps stay gaps), and the transit folded."""
    series = {"t": [], "y_ppm": [], "sector": []}
    for s in np.unique(lc["sector"]):
        m = lc["sector"] == s
        t = lc["t"][m]
        tb, yb, _ = binned(t, lc["y"][m], lc["w"][m], np.arange(t.min(), t.max() + BIN_D, BIN_D))
        series["t"] += np.round(tb, 4).tolist()
        series["y_ppm"] += np.round(yb * 1e6).astype(int).tolist()
        series["sector"] += [int(s)] * len(tb)
    x = phase_offset(lc["t"], sig["period"], sig["t0"])
    half = FOLD_HALF * sig["dur"]
    near = np.abs(x) < half
    xb, yb, eb = binned(x[near], lc["y"][near], lc["w"][near], np.linspace(-half, half, FOLD_BINS + 1))
    return {"series": series, "fold": {"x_h": np.round(xb * 24, 3).tolist(), "y_ppm": np.round(yb * 1e6, 1).tolist(),
                                       "err_ppm": np.round(eb * 1e6, 1).tolist()}}


def dossier(tic: int, model: str, prompt: str) -> dict | None:
    path = DATA / "dossiers" / f"{tic}.{model}.{prompt}.json"
    if not path.exists():
        return None
    run = json.loads(path.read_text(encoding="utf-8"))
    return {**run["dossier"], "model": run["model"], "prompt": run["prompt"], "cost_usd": run["cost"]}


def export(out: Path, model: str, prompt: str) -> dict:
    (out / "stars").mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(DATA / "targets.parquet", columns=["tic", "ra", "dec", "tmag", "teff", "rad", "two_min_sectors"])
    status = pd.Series(STATUS["pool"], index=pool["tic"])
    held = {int(t) for t in (ROOT / "survey" / "holdout.txt").read_text(encoding="utf-8").split()}
    status[status.index.isin(held)] = STATUS["held back"]

    counts = {}
    for name, results, vet_dir, lc_dir in RUNS:
        if not results.exists():
            continue
        res = pd.read_csv(results)
        status[status.index.isin(res["tic"])] = STATUS["searched"]
        final = {}
        if (vet_dir / "status.csv").exists():
            with (vet_dir / "status.csv").open(newline="", encoding="utf-8") as f:
                final = {int(r["tic"]): r for r in csv.DictReader(f)}
        for _, r in res[res["gate1"] == "pass"].iterrows():
            tic = int(r["tic"])
            st = final.get(tic, {"status": "awaiting agent", "reasons": ""})
            status[tic] = STATUS[st["status"]]
            counts[st["status"]] = counts.get(st["status"], 0) + 1
            checks = json.loads((vet_dir / f"{tic}.json").read_text(encoding="utf-8"))
            sig = {k: checks["signal"][k] for k in ("period", "t0", "dur")}
            raw = load_lc(tic, lc_dir)
            page = {"tic": tic, "batch": name, "status": st["status"],
                    "reasons": [x for x in st["reasons"].split("; ") if x],
                    "gate1": {"dip_snr": r["dip_snr"], "flipped_snr": r["bump_snr"], "transits": int(r["dip_transits"])},
                    "checks": checks, "dossier": dossier(tic, model, prompt),
                    **light_curve(detrend(raw["time"], raw["flux"], raw["sector"]), sig)}
            cand = vet_dir / "candidates" / f"{tic}.json"
            if cand.exists():
                c = json.loads(cand.read_text(encoding="utf-8"))
                page["novelty"], page["when"] = c.get("novelty"), c.get("when")
            (out / "stars" / f"{tic}.json").write_text(json.dumps(page, separators=(",", ":"), default=float),
                                                         encoding="utf-8")
        gate = vet_dir.parent.parent / f"{name}_gate1.json"
        counts[f"{name}_searched"] = int(len(res))
        if gate.exists():
            counts[f"{name}_gate1"] = json.loads(gate.read_text(encoding="utf-8"))

    sky = pool.assign(status=pool["tic"].map(status).astype(int))
    (out / "sky.json").write_text(json.dumps({
        "tic": sky["tic"].tolist(), "ra": sky["ra"].round(4).tolist(), "dec": sky["dec"].round(4).tolist(),
        "tmag": sky["tmag"].round(2).tolist(), "teff": sky["teff"].round(0).astype(int).tolist(),
        "status": sky["status"].tolist(), "codes": STATUS}, separators=(",", ":")), encoding="utf-8")
    summary = {"pool": len(pool), "funnel": json.loads((DATA / "targets_funnel.json").read_text(encoding="utf-8")),
               "held_back": len(held), "by_status": {k: int((sky["status"] == v).sum()) for k, v in STATUS.items()},
               **counts}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA / "site"))
    ap.add_argument("--model", default="muse-spark-1.3-contributor")
    ap.add_argument("--prompt", default="v5")
    args = ap.parse_args()
    print(json.dumps(export(Path(args.out), args.model, args.prompt), indent=1, default=float))


if __name__ == "__main__":
    main()
