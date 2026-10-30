"""Fast TRICERATOPS (pipeline/tri_fast.py) on a list of signals, one run each, same seed, a result per star as it finishes.

Kaggle (CPU): --jobs fpp_jobs.json folds each signal from the mounted fetch outputs, writes fpp_input_TIC.json, runs it.
Laptop (TRICERATOPS venv): --folded DIR runs the fpp_input_TIC.json files already there (the Kaggle job's), with a setup cache.
Per star it records FPP, NFPP, class, setup seconds (apertures, catalogues, calc_depths) and calc seconds (calc_probs), or
why the star is uncovered, in results.json beside the per-star fpp_TIC.json. Stops starting stars after --max-minutes.

Run: E:/AI/_ref/tri-venv/Scripts/python tools/fpp_survivors.py --folded DIR --out OUT --cache CACHE --max-minutes 30
Kaggle: this file is the job's entry; pipeline/ and finetune/ sit beside it (see tools/kaggle-run/README.md).
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]  # repo root on the laptop; the job folder on Kaggle

import numpy as np  # noqa: E402

from pipeline import fpp, tri_fast  # noqa: E402

SEED = 0
TIMES = {}


def install_timers() -> None:
    """Seed before each calc_probs and time it; setup is the rest of run_one."""
    import triceratops.triceratops as tr
    probs = tr.target.calc_probs

    def calc_probs(self, *a, **k):
        np.random.seed(SEED)
        t = time.perf_counter()
        try:
            return probs(self, *a, **k)
        finally:
            TIMES["calc"] = time.perf_counter() - t

    tr.target.calc_probs = calc_probs


def write_all(rows: list[dict], out: Path) -> None:
    (out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=Path)
    ap.add_argument("--folded", type=Path)
    ap.add_argument("--inputs", type=Path, default=Path(os.environ.get("KAGGLE_INPUTS", "/kaggle/input")))
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--only", type=int, nargs="+")
    args = ap.parse_args()
    args.runs = 1
    kaggle = Path("/kaggle/working").exists()
    out = args.out or (Path("/kaggle/working") if kaggle else Path("fpp_survivors_out"))
    out.mkdir(parents=True, exist_ok=True)
    cache = args.cache or (Path("/kaggle/temp/setup_cache") if kaggle else None)
    scratch = Path("/kaggle/temp/fpp") if kaggle else None  # attempt() clears lightkurve's cache when given one: not on the laptop
    if scratch:
        scratch.mkdir(parents=True, exist_ok=True)
    fpp.retrying_downloads()
    deadline = fpp.T0 + args.max_minutes * 60
    rows, why = [], {}
    if args.jobs:
        jobs = json.loads(args.jobs.read_text(encoding="utf-8"))[args.part::args.parts]
        found = fpp.folds(jobs, args.inputs, out)
        for tic, d in found.items():
            (out / f"fpp_input_{tic}.json").write_text(json.dumps(d), encoding="utf-8")
        order = [j["tic"] for j in jobs]
        for r in json.loads((out / "coverage.json").read_text(encoding="utf-8")):
            if not r["covered"]:
                why[r["tic"]] = r["why"]
    else:
        found = {}
        for p in sorted(args.folded.glob("fpp_input_*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            found[d["tic"]] = d
        order = list(found)
    if args.only:
        order = [t for t in order if t in args.only]
    order = order[:args.limit]
    import numba
    env = {"machine": "kaggle" if kaggle else "laptop", "cpu_count": os.cpu_count(), "numba": numba.__version__,
           "numba_threads": numba.config.NUMBA_NUM_THREADS, "platform": platform.platform(), "seed": SEED, "n": args.n}
    print(env, flush=True)
    tri_fast.install(cache)
    install_timers()
    fpp.N_SIM = args.n
    for n, tic in enumerate(order, 1):
        row = {"tic": tic}
        if tic not in found:
            row |= {"covered": False, "why": why.get(tic, "no folded input")}
        elif time.time() > deadline:
            fpp.log("time limit: no more stars")
            row |= {"covered": True, "ran": False, "why": "time limit"}
        else:
            TIMES.clear()
            t = time.perf_counter()
            fpp.attempt(found[tic], args, deadline, out, scratch)
            wall = time.perf_counter() - t
            r = json.loads((out / f"fpp_{tic}.json").read_text(encoding="utf-8"))
            if r.get("FPP_mean") is None:
                row |= {"covered": True, "ran": False, "why": r.get("error", "no result"), "wall_s": round(wall, 1)}
            else:
                calc = TIMES.get("calc", 0.0)
                row |= {"covered": True, "ran": True, "FPP": r["FPP_mean"], "NFPP": r["NFPP_mean"], "class": r["class"],
                        "aperture": r["aperture"], "n_bins": r["n_bins"], "setup_s": round(wall - calc, 1),
                        "calc_s": round(calc, 1), "wall_s": round(wall, 1)}
        rows.append(row)
        write_all(rows, out)
        fpp.log(f"{n}/{len(order)} {json.dumps(row)}")
    calcs = [r["calc_s"] for r in rows if r.get("ran")]
    fpp.log(f"{len(calcs)} ran, median calc {statistics.median(calcs) if calcs else 0:.1f} s, "
            f"total {sum(r.get('wall_s', 0) for r in rows) / 60:.1f} min")


if __name__ == "__main__":
    main()
