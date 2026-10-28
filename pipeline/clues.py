"""System 1's clues for signals vetted before System 1 was in the pipeline: a Kaggle CPU job.

For every gate 1 pass: the centroid where the code rules skipped it, the gaps check where the vet file
predates it, and the transit fit; then Gaia's binarity signs for every target and Gaia's eclipsing-binary
periods for the targets and their suspects. The same clues finetune/clues.py made for the labelled TOIs.

export (local): jobs.json from the vet files (Gaia IDs exact, from the target table), plus the code, in
  a small job folder (mobile data drops uploads over ~1 MB; the light curves are mounted instead).
run (Kaggle, with tessa-fetch and tessa-fetch2 mounted): writes clues/TIC.json and gaia.json.

Run: python -m pipeline.clues export data/vet/survey data/vet/batch2 --out DIR
     Kaggle: python s1_clues.py [--workers 6] [--max-minutes 300] [--limit N]
Then: python -m pipeline.system1 data/vet/survey data/vet/batch2 --clues DIR_WITH_OUTPUT
"""
import argparse
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
STAR_KEYS = ("tic", "source_id", "tmag", "teff", "rad", "mass", "distance", "contratio", "ruwe",
             "phot_variable_flag", "non_single_star", "ra", "dec", "mass_estimated")
ENTRY = "from pipeline.clues import main\n\nif __name__ == \"__main__\":  # spawn re-imports this file\n    main()\n"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min  {msg}", flush=True)


# --- export (local) ----------------------------------------------------------------------------

def export(vet_dirs: list[Path], out: Path) -> None:
    import pandas as pd
    from .paths import DATA, ROOT
    from .system1 import suspects
    ids = pd.read_parquet(DATA / "targets.parquet", columns=["tic", "source_id"]).dropna()
    exact = dict(zip(ids["tic"].astype("int64"), ids["source_id"].astype("int64")))  # never through float
    jobs = []
    for vet_dir in vet_dirs:
        for path in sorted(vet_dir.glob("[0-9]*.json")):
            c = json.loads(path.read_text(encoding="utf-8"))
            star = {k: c["star"][k] for k in STAR_KEYS if k in c["star"]}
            star["source_id"] = int(exact[c["tic"]]) if c["tic"] in exact else None
            jobs.append({"tic": c["tic"], "batch": vet_dir.name, "signal": c["signal"], "star": star,
                         "need_centroid": not c.get("centroid"), "need_gaps": "gaps" not in c,
                         "suspects": sorted(suspects(c))})
    (out / "pipeline").mkdir(parents=True, exist_ok=True)
    for p in (ROOT / "pipeline").glob("*.py"):
        shutil.copy(p, out / "pipeline" / p.name)
    (out / "jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    (out / "s1_clues.py").write_text(ENTRY, encoding="utf-8")
    (out / "requirements.txt").write_text("astroquery==0.4.11\n", encoding="utf-8")
    print(f"{len(jobs)} signals: {sum(j['need_centroid'] for j in jobs)} need a centroid, "
          f"{sum(j['need_gaps'] for j in jobs)} the gaps check; {sum(j['star']['source_id'] is None for j in jobs)} without a Gaia ID")


# --- run (Kaggle) ------------------------------------------------------------------------------

def shards_by_tic(root: Path) -> dict[int, str]:
    import csv
    found = {}
    for manifest in root.rglob("manifest.csv"):
        with manifest.open(newline="") as f:
            for r in csv.DictReader(f):
                if int(r["n_sectors"]):
                    found[int(r["tic"])] = str(manifest.parent / f"lc_{int(r['shard']):03d}.npz")
    return found


def one(job: dict, shard: str, out: str) -> dict:
    from .fit import transit_fit
    from .search import detrend
    from .vet import finite, gaps
    tic, sig, star = job["tic"], job["signal"], job["star"]
    t_start = time.time()
    res = {"tic": tic}
    try:
        with np.load(shard) as z:
            raw = {k: z[f"{tic}_{k}"] for k in ("time", "flux", "sector", "source")}
        if job["need_gaps"]:
            res["gaps"] = gaps(raw, sig)
        if job["need_centroid"]:
            from .centroid import where
            res["centroid"] = where(tic, sig, star=star, sectors=sorted({int(s) for s in raw["sector"]}))
        res["fit"] = transit_fit(detrend(raw["time"], raw["flux"], raw["sector"]), sig, star)
    except Exception as e:  # one bad star must not stop the job
        res["error"] = f"{e.__class__.__name__}: {e}"[:300]
    res["seconds"] = round(time.time() - t_start, 1)
    Path(out, "clues", f"{tic}.json").write_text(json.dumps(finite(res), default=float), encoding="utf-8")
    return res


def main() -> None:
    from .fit import gaia_ebs, gaia_extra
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", default="jobs.json")
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--inputs", default=os.environ.get("KAGGLE_INPUTS", "/kaggle/input"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-minutes", type=float, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "clues").mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))[:args.limit]
    shards = shards_by_tic(Path(args.inputs))
    todo = [j for j in jobs if j["tic"] in shards and not (out / "clues" / f"{j['tic']}.json").exists()]
    log(f"{len(jobs)} signals, {len(shards)} light curves mounted, {len(todo)} to do "
        f"({sum(j['need_centroid'] for j in todo)} centroids)")

    ids = sorted({j["star"]["source_id"] for j in jobs if j["star"].get("source_id")})
    extra = gaia_extra(ids)
    (out / "gaia.json").write_text(json.dumps({"extra": extra, "eb": {}}), encoding="utf-8")

    found = {}
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = {pool.submit(one, j, shards[j["tic"]], str(out)): j["tic"] for j in todo}
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            found[r["tic"]] = r
            if n % 25 == 0:
                log(f"{n}/{len(todo)}, errors {sum('error' in x for x in found.values())}")
            if (time.time() - T0) / 60 > args.max_minutes:
                log("time limit reached; cancelling the rest")
                for g in futures:
                    g.cancel()
                break

    near = set(ids)
    for j in jobs:
        near |= set(j["suspects"])
        cen = found.get(j["tic"], {}).get("centroid") or {}
        near |= {int(s["source_id"]) for s in cen.get("suspects") or [] if s.get("source_id")}
    (out / "gaia.json").write_text(json.dumps({"extra": extra, "eb": gaia_ebs(sorted(near))}), encoding="utf-8")
    log(f"done: {len(found)} signals, {sum('error' in r for r in found.values())} errors, "
        f"{sum((r.get('fit') or {}).get('status') == 'ok' for r in found.values())} fits ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("vet_dirs", nargs="+", type=Path)
        ap.add_argument("--out", type=Path, required=True)
        a = ap.parse_args()
        export(a.vet_dirs, a.out)
    else:
        main()
