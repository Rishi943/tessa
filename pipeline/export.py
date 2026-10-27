"""Export the survey for the site.

sky.json has every star in the pool, column by column, with a status code; summary.json has the
funnel and the counts; stars/TIC.json has, for every star that passed gate 1, its light curve
(binned and folded), the code checks, the agent's dossier, the gate 2 status (with the case-closed line
for rejected stars and the tools' evidence for investigated ones) and, for candidates, the novelty
check and the predictions.

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
    """The investigation's dossier when the agent investigated this star, else the vetting one."""
    for version in ("inv1", prompt):
        path = DATA / "dossiers" / f"{tic}.{model}.{version}.json"
        if path.exists():
            run = json.loads(path.read_text(encoding="utf-8"))
            return {**run["dossier"], "model": run["model"], "prompt": run["prompt"], "cost_usd": run["cost"],
                    "tools_called": [t["tool"] for t in run.get("tools", [])], "invented": run.get("invented", [])}
    return None


def dip_row(page: dict) -> dict:
    """One line per gate-1 pass, for the site's list of dips and Ask Tessa's search."""
    c, d = page["checks"], page["dossier"] or {}
    r = lambda v, n=None: None if v is None else round(v, n)  # Gaia has gaps
    return {"tic": page["tic"], "batch": page["batch"], "status": page["status"],
            "period_d": r(c["signal"]["period"], 5), "depth_ppm": r(c["signal"]["depth_ppm"]),
            "snr": r(c["signal"]["snr"], 1), "rp_rearth": r(c["size"]["rp_rearth"], 2),
            "tmag": r(c["star"]["tmag"], 2), "teff": r(c["star"]["teff"]), "dist_pc": r(c["star"]["distance"], 1),
            "ra": r(c["star"]["ra"], 4), "dec": r(c["star"]["dec"], 4),
            "line": page["case_closed"] or d.get("public_summary") or "; ".join(page["reasons"]),
            "reasons": page["reasons"], "dossier": bool(d), "tools": len(d.get("tools_called", []))}


def export(out: Path, model: str, prompt: str) -> dict:
    (out / "stars").mkdir(parents=True, exist_ok=True)
    pool = pd.read_parquet(DATA / "targets.parquet", columns=["tic", "ra", "dec", "parallax", "tmag", "teff", "rad",
                                                              "two_min_sectors"])
    status = pd.Series(STATUS["pool"], index=pool["tic"])
    held = {int(t) for t in (ROOT / "survey" / "holdout.txt").read_text(encoding="utf-8").split()}
    status[status.index.isin(held)] = STATUS["held back"]

    counts = {}
    dips = []
    costs = {"vetting": 0.0, "investigation": 0.0, "case_closed": 0.0, "observing": 0.0, "audit": 0.0}
    invented = {"dossiers": 0, "with_invented_numbers": 0}  # numbers a model wrote that were not in its input
    for name, results, vet_dir, lc_dir in RUNS:
        if not results.exists():
            continue
        for step, csv_name in (("vetting", f"dossiers.{model}.{prompt}.csv"), ("investigation", f"investigations.{model}.inv1.csv")):
            if (vet_dir / csv_name).exists():
                runs = pd.read_csv(vet_dir / csv_name)
                costs[step] += float(runs["cost"].fillna(0).sum())
                invented["dossiers"] += len(runs)
                invented["with_invented_numbers"] += int((runs["invented"].fillna(0) > 0).sum())
        if (vet_dir / "caseclosed.meta.json").exists():
            costs["case_closed"] += json.loads((vet_dir / "caseclosed.meta.json").read_text(encoding="utf-8"))["cost_usd"]
        res = pd.read_csv(results)
        status[status.index.isin(res["tic"])] = STATUS["searched"]
        final = {}
        closed = json.loads((vet_dir / "caseclosed.json").read_text(encoding="utf-8"))             if (vet_dir / "caseclosed.json").exists() else {}
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
            cand = vet_dir / "candidates" / f"{tic}.json"
            c = json.loads(cand.read_text(encoding="utf-8")) if cand.exists() else {}
            eph = (c.get("when") or {}).get("ephemeris") or {}
            if eph.get("ok"):  # the refitted ephemeris centres the fold better than the search grid
                sig = {**sig, "period": eph["period"], "t0": eph["t0_btjd"]}
            raw = load_lc(tic, lc_dir)
            page = {"tic": tic, "batch": name, "status": st["status"],
                    "reasons": [x for x in st["reasons"].split("; ") if x],
                    "gate1": {"dip_snr": r["dip_snr"], "flipped_snr": r["bump_snr"], "transits": int(r["dip_transits"])},
                    "case_closed": closed.get(str(tic)), "checks": checks, "dossier": dossier(tic, model, prompt),
                    "ephem": {"period": sig["period"], "t0": sig["t0"]},
                    **light_curve(detrend(raw["time"], raw["flux"], raw["sector"]), sig)}
            ev = DATA / "evidence" / f"{tic}.json"
            if ev.exists():
                page["evidence"] = json.loads(ev.read_text(encoding="utf-8"))
            if c:
                page["novelty"], page["when"], page["observing"] = c.get("novelty"), c.get("when"), c.get("observing")
                costs["observing"] += (c.get("observing") or {}).get("cost_usd", 0)
                audit = DATA / "dossiers" / f"{tic}.muse-wonder-1.0.{prompt}.json"  # second opinion, shortlist only
                if audit.exists():
                    run = json.loads(audit.read_text(encoding="utf-8"))
                    page["audit"] = {**run["dossier"], "model": run["model"], "cost_usd": run["cost"]}
                    costs["audit"] += run["cost"]
            (out / "stars" / f"{tic}.json").write_text(json.dumps(page, separators=(",", ":"), default=float),
                                                         encoding="utf-8")
            dips.append(dip_row(page))
        gate = vet_dir.parent.parent / f"{name}_gate1.json"
        counts[f"{name}_searched"] = int(len(res))
        if gate.exists():
            counts[f"{name}_gate1"] = json.loads(gate.read_text(encoding="utf-8"))

    sky = pool.assign(status=pool["tic"].map(status).astype(int))
    (out / "sky.json").write_text(json.dumps({
        "tic": sky["tic"].tolist(), "ra": sky["ra"].round(4).tolist(), "dec": sky["dec"].round(4).tolist(),
        "dist": (1000 / sky["parallax"]).round(2).tolist(), "tmag": sky["tmag"].round(2).tolist(),
        "teff": sky["teff"].round(0).astype(int).tolist(), "status": sky["status"].tolist(), "codes": STATUS},
        separators=(",", ":")), encoding="utf-8")
    (out / "dips.json").write_text(json.dumps(dips, separators=(",", ":"), default=float), encoding="utf-8")
    # Any star by TIC, in 100 small files, so Ask Tessa can say what happened to a star without a case file
    (out / "pool").mkdir(exist_ok=True)
    names = {v: k for k, v in STATUS.items()}
    for shard, part in sky.groupby(sky["tic"] % 100):
        rows = {int(r.tic): {"status": names[r.status], "dist_pc": round(1000 / r.parallax, 1), "tmag": round(r.tmag, 2),
                             "teff": None if pd.isna(r.teff) else int(r.teff), "ra": round(r.ra, 4), "dec": round(r.dec, 4)}
                for r in part.itertuples()}
        (out / "pool" / f"{shard}.json").write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    summary = {"pool": len(pool), "funnel": json.loads((DATA / "targets_funnel.json").read_text(encoding="utf-8")),
               "held_back": len(held), "by_status": {k: int((sky["status"] == v).sum()) for k, v in STATUS.items()},
               "costs_usd": {k: round(v, 4) for k, v in costs.items()}, "invented": invented,
               "dossiers": sum(d["dossier"] for d in dips), "investigated": sum(d["tools"] > 0 for d in dips),
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
