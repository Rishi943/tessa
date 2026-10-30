"""Re-vet the survey and batch 2 on Kaggle CPU, with the production vet path (pipeline/vet.vet) unchanged.

The slow part of vetting is TESScut and Gaia, and it needs only the light curves (mounted from the fetch jobs)
and a few small tables, which go up as a private dataset.

export (local): python -m pipeline.revet export --out JOB_DIR --support SUPPORT_DIR [--skip VET_DIR ...]
  JOB_DIR     the code (pipeline/*.py), jobs.json (every gate 1 pass of both searches except the --skip stars) and the entry
  SUPPORT_DIR the dataset: targets.parquet cut to the jobs' stars, their Gaia neighbour files, the catalog lists
              vet reads, and System 1's model files (nothing else: no credentials, no cutouts)
run (Kaggle, tessa-fetch and tessa-fetch2 mounted, plus the support dataset): s1_revet.py --part I --parts N
  writes /kaggle/working/vet/BATCH/TIC.json and /kaggle/working/evidence/TIC.json as each star finishes, and
  status.json (done, no light curve, errors). Stars already in a mounted earlier output are skipped, so a part
  that hit its time cap resumes by mounting it with --kernel.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

T0 = time.time()
RESULTS = {"survey": "survey_results.csv", "batch2": "batch2_results.csv"}
ENTRY = "from pipeline.revet import main\n\nif __name__ == \"__main__\":  # spawn re-imports this file\n    main()\n"
CATALOG_FILES = ("toi.csv", "ctoi.csv", "confirmed.csv", "tess_ebs.csv", "planets.csv", "planets_orbits.csv",
                 "open_tois.csv", "open_tois_orbits.csv", "reported.csv")
SYSTEM1_FILES = ("xgb-clues.json", "calibration-clues.json", "results-clues.json", "preds-clues.json",
                 "similar-clues.npz", "similar-clues.json", "params.json")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f} min  {msg}", flush=True)


# --- export (local) ----------------------------------------------------------------------------

def export(out: Path, support: Path, skip_dirs: list[Path]) -> None:
    import pandas as pd
    from .paths import CACHE, CATALOGS, DATA, ROOT
    from .vet import signal_from_results
    skip = {int(p.stem) for d in skip_dirs for p in d.glob("[0-9]*.json")}
    jobs = []
    for batch, name in RESULTS.items():
        with (DATA / name).open(newline="") as f:
            tics = [int(r["tic"]) for r in csv.DictReader(f) if r["gate1"] == "pass"]
        jobs += [{"tic": t, "batch": batch, "signal": signal_from_results(t, DATA / name)} for t in tics if t not in skip]
    tics = sorted({j["tic"] for j in jobs})
    (out / "pipeline").mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "pipeline").glob("*.py"):
        shutil.copy(p, out / "pipeline" / p.name)
    (out / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    (out / "s1_revet.py").write_text(ENTRY, encoding="utf-8")
    import xgboost
    (out / "requirements.txt").write_text(f"astroquery==0.4.11\nxgboost=={xgboost.__version__}\n", encoding="utf-8")

    support.mkdir(parents=True, exist_ok=True)
    pd.read_parquet(DATA / "targets.parquet").query("tic in @tics").to_parquet(support / "targets.parquet")
    (support / "gaia_neighbours").mkdir(exist_ok=True)
    for t in tics:
        src = CACHE / "gaia_neighbours" / f"{t}.csv"
        if src.exists():
            shutil.copy(src, support / "gaia_neighbours" / src.name)
    (support / "catalogs").mkdir(exist_ok=True)
    for name in CATALOG_FILES:
        shutil.copy(CATALOGS / name, support / "catalogs" / name)
    (support / "system1").mkdir(exist_ok=True)
    for name in SYSTEM1_FILES:
        shutil.copy(DATA / "finetune" / "system1" / name, support / "system1" / name)
    shutil.copy(DATA / "finetune" / "labels.json", support / "labels.json")
    shutil.copy(DATA / "fetch_manifest.csv", support / "fetch_manifest.csv")  # the survey's, where centroid looks first
    size = sum(p.stat().st_size for p in support.rglob("*") if p.is_file())
    print(f"{len(jobs)} signals to vet ({len(skip)} skipped as done), {len(tics)} stars; support dataset {size / 1e6:.1f} MB, "
          f"{len(list((support / 'gaia_neighbours').iterdir()))} neighbour files")


# --- run (Kaggle) ------------------------------------------------------------------------------

def shards_by_tic(root: Path) -> dict[int, str]:
    found = {}
    for manifest in root.rglob("manifest.csv"):
        with manifest.open(newline="") as f:
            for r in csv.DictReader(f):
                if int(r["n_sectors"]):
                    found[int(r["tic"])] = str(manifest.parent / f"lc_{int(r['shard']):03d}.npz")
    return found


def setup(inputs: Path) -> None:
    """Put the support dataset where pipeline.paths looks, and the light curves' manifests where centroid looks."""
    from .paths import CACHE, CATALOGS, DATA
    root = next(p.parent for p in inputs.rglob("targets.parquet"))
    shutil.copy(root / "targets.parquet", DATA / "targets.parquet")
    shutil.copytree(root / "gaia_neighbours", CACHE / "gaia_neighbours", dirs_exist_ok=True)
    shutil.copytree(root / "catalogs", CATALOGS, dirs_exist_ok=True)
    (DATA / "finetune").mkdir(exist_ok=True)
    shutil.copytree(root / "system1", DATA / "finetune" / "system1", dirs_exist_ok=True)
    shutil.copy(root / "labels.json", DATA / "finetune" / "labels.json")
    shutil.copy(root / "fetch_manifest.csv", DATA / "fetch_manifest.csv")
    for i, manifest in enumerate(sorted(inputs.rglob("manifest.csv"))):
        (CACHE / f"lc_k{i}").mkdir(exist_ok=True)
        shutil.copy(manifest, CACHE / f"lc_k{i}" / "manifest.csv")


def one(job: dict, shard: str, out: str) -> dict:
    from .paths import DATA
    from .vet import finite, vet
    tic = job["tic"]
    res = {"tic": tic, "batch": job["batch"]}
    t_start = time.time()
    try:
        with np.load(shard) as z:
            raw = {k: z[f"{tic}_{k}"] for k in ("time", "flux", "sector", "source")}
        d = vet(tic, job["signal"], Path("."), raw=raw)
        folder = Path(out, "vet", job["batch"])
        folder.mkdir(parents=True, exist_ok=True)
        ev = DATA / "evidence" / f"{tic}.json"
        if ev.exists():
            Path(out, "evidence").mkdir(exist_ok=True)
            shutil.copy(ev, Path(out, "evidence", ev.name))
        (folder / f"{tic}.json").write_text(json.dumps(finite(d), indent=1, default=float), encoding="utf-8")  # last: marks it done
        res["verdict"] = d["code_verdict"]
    except Exception as e:  # one bad star must not stop the job; nothing is saved for it
        res["error"] = f"{e.__class__.__name__}: {e}"[:300]
    res["seconds"] = round(time.time() - t_start, 1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", default="jobs.json")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--inputs", default=os.environ.get("KAGGLE_INPUTS", "/kaggle/input"))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tics", help="only these stars, comma separated")
    ap.add_argument("--max-minutes", type=float, default=660)
    args = ap.parse_args()
    out, inputs = Path(args.out), Path(args.inputs)
    setup(inputs)
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    if args.tics:
        keep = {int(t) for t in args.tics.split(",")}
        jobs = [j for j in jobs if j["tic"] in keep]
    jobs = jobs[args.part::args.parts][:args.limit]
    shards = shards_by_tic(inputs)
    done = {int(p.stem) for p in inputs.rglob("vet/*/[0-9]*.json")} | {int(p.stem) for p in out.glob("vet/*/[0-9]*.json")}
    no_lc = [j["tic"] for j in jobs if j["tic"] not in shards]
    todo = [j for j in jobs if j["tic"] in shards and j["tic"] not in done]
    log(f"part {args.part}/{args.parts}: {len(jobs)} signals, {len(shards)} light curves mounted, {len(no_lc)} without one, "
        f"{len(jobs) - len(no_lc) - len(todo)} already done, {len(todo)} to do")

    found = {}

    def status(final: bool) -> None:
        s = {"part": args.part, "parts": args.parts, "final": final, "jobs": len(jobs), "todo": len(todo), "no_light_curve": no_lc,
             "vetted": sum("error" not in r for r in found.values()),
             "errors": {r["tic"]: r["error"] for r in found.values() if "error" in r},
             "verdicts": {v: sum(r.get("verdict") == v for r in found.values()) for v in ("reject", "agent")},
             "minutes": round((time.time() - T0) / 60, 1)}
        (out / "status.json").write_text(json.dumps(s, indent=1), encoding="utf-8")

    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(one, j, shards[j["tic"]], str(out)): j["tic"] for j in todo}
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            found[r["tic"]] = r
            if "error" in r:
                log(f"{r['tic']}: {r['error']}")
            if n % 10 == 0:
                log(f"{n}/{len(todo)}, errors {sum('error' in x for x in found.values())}")
                status(False)
            if (time.time() - T0) / 60 > args.max_minutes:
                log("time limit reached; cancelling the rest (mount this output with --kernel to resume)")
                for g in futures:
                    g.cancel()
                break
    status(True)
    log(f"done: {len(found)} vetted or failed, {sum('error' in r for r in found.values())} errors, {len(no_lc)} without light curve")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--out", type=Path, required=True)
        ap.add_argument("--support", type=Path, required=True)
        ap.add_argument("--skip", type=Path, nargs="*", default=[])
        a = ap.parse_args()
        export(a.out, a.support, a.skip)
    else:
        main()
