"""Where does stock TRICERATOPS spend its time on one signal? Timers around pipeline.fpp.run_one, results unchanged.

Timed: setup (apertures, tr.target, calc_depths; done once and reused for every N), calc_probs, every
triceratops.marginal_likelihoods.lnZ_* (seconds and calls, by name), lnL_TP_p / lnL_EB_p / lnL_EB_twin_p (the light
curve functions the scenarios call, patched in marginal_likelihoods), and numba's JIT cost (first call of each
lnL_ function minus the median of its later calls, so it only shows in the first N of a process).
The numpy seed is fixed before each calc_probs. Nothing is committed or installed; the library is not edited.

Run (laptop, TRICERATOPS venv):  E:/AI/_ref/tri-venv/Scripts/python tools/time_fpp.py --input fpp_input_TIC.json --out DIR
Run (Kaggle): the job folder holds this file as the entry plus pipeline/ and finetune/; --jobs fpp_jobs.json folds the
  signal from the mounted light curves, writes fpp_input_TIC.json (for the laptop) and times it.
"""
import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]  # repo root on the laptop; the job folder on Kaggle

import numpy as np  # noqa: E402

from pipeline import fpp  # noqa: E402

LNL = ("lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p")
SEED = 0


class Timers:
    def __init__(self):
        self.reset()

    def reset(self):
        self.calls = defaultdict(list)  # name -> [seconds per call], in call order
        self.setup = {}
        self.calc_probs = 0.0


T = Timers()


def timed(name, fn):
    def wrapper(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T.calls[name].append(time.perf_counter() - t)
    wrapper.__name__ = name
    return wrapper


def install():
    """Patch the timers in. Returns the stock target class and its methods for the setup steps."""
    import triceratops.marginal_likelihoods as ml
    import triceratops.triceratops as tr
    for name in LNL:
        setattr(ml, name, timed(name, getattr(ml, name)))
    for name in [n for n in dir(ml) if n.startswith("lnZ_")]:
        wrapped = timed(name, getattr(ml, name))
        setattr(ml, name, wrapped)
        setattr(tr, name, wrapped)  # triceratops.py star-imported the names, so calc_probs looks them up here

    cls = tr.target
    made = {}

    def target(*a, **k):  # setup once, reused by the next N (same inputs, so the same object state)
        if "target" not in made:
            t = time.perf_counter()
            made["target"] = cls(*a, **k)
            T.setup["tr.target"] = time.perf_counter() - t
        return made["target"]

    depths, probs = cls.calc_depths, cls.calc_probs

    def calc_depths(self, *a, **k):
        if "calc_depths" not in T.setup:
            t = time.perf_counter()
            depths(self, *a, **k)
            T.setup["calc_depths"] = time.perf_counter() - t

    def calc_probs(self, *a, **k):
        np.random.seed(SEED)
        T.calls.clear()
        t = time.perf_counter()
        try:
            return probs(self, *a, **k)
        finally:
            T.calc_probs = time.perf_counter() - t

    target.__wrapped__ = cls
    tr.target = target
    cls.calc_depths, cls.calc_probs = calc_depths, calc_probs

    real_apertures = fpp.apertures

    def apertures(*a, **k):
        if "apertures" not in T.setup:
            t = time.perf_counter()
            made["aps"] = real_apertures(*a, **k)
            T.setup["apertures"] = time.perf_counter() - t
        return made["aps"]

    fpp.apertures = apertures


def summarise(n: int, res: dict, wall: float) -> dict:
    z = {k: v for k, v in T.calls.items() if k.startswith("lnZ_")}
    lz = {k: {"seconds": round(sum(v), 3), "calls": len(v)} for k, v in sorted(z.items(), key=lambda kv: -sum(kv[1]))}
    ll = {k: {"seconds": round(sum(T.calls[k]), 3), "calls": len(T.calls[k])} for k in LNL}
    jit = {k: round(max(0.0, v[0] - statistics.median(v[1:])), 3) for k, v in T.calls.items() if k in LNL and len(v) > 2}
    z_total, l_total = sum(sum(v) for v in z.values()), sum(sum(T.calls[k]) for k in LNL)
    first_z = next(iter(z.items()), None)
    return {"N": n, "FPP": res["FPP_mean"], "NFPP": res["NFPP_mean"], "wall_seconds": round(wall, 1),
            "calc_probs_seconds": round(T.calc_probs, 2), "lnZ_total_seconds": round(z_total, 2),
            "lnL_total_seconds": round(l_total, 2), "scenario_rest_seconds": round(z_total - l_total, 2),
            "outside_lnZ_seconds": round(T.calc_probs - z_total, 2), "lnZ": lz, "lnL": ll,
            "jit_extra_seconds_by_lnL": jit, "jit_extra_seconds": round(sum(jit.values()), 2),
            "scenarios": res["runs"][0]["scenarios"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, help="a folded fpp_input_TIC.json")
    ap.add_argument("--jobs", type=Path, help="fpp_jobs.json with the one signal: fold it from the mounted light curves")
    ap.add_argument("--inputs", type=Path, default=Path(os.environ.get("KAGGLE_INPUTS", "/kaggle/input")))
    ap.add_argument("--ns", type=int, nargs="+", default=[1_000_000, 100_000])
    ap.add_argument("--fast", action="store_true", help="the fast likelihoods (pipeline/tri_fast.py) instead of stock's")
    ap.add_argument("--cache", type=Path, help="with --fast: keep each star's setup here")
    ap.add_argument("--machine", default="kaggle" if Path("/kaggle/working").exists() else "laptop")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("time_fpp_out"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fpp.retrying_downloads()
    if args.jobs:
        jobs = json.loads(args.jobs.read_text(encoding="utf-8"))
        d = fpp.folds(jobs, args.inputs, args.out)[jobs[0]["tic"]]
        (args.out / f"fpp_input_{d['tic']}.json").write_text(json.dumps(d), encoding="utf-8")
    else:
        d = json.loads(args.input.read_text(encoding="utf-8"))
    import numba
    import triceratops
    env = {"machine": args.machine, "cpu_count": os.cpu_count(),
           "cpus_usable": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
           "numba_threads": numba.config.NUMBA_NUM_THREADS, "numba_threading_layer_env": os.environ.get("NUMBA_THREADING_LAYER"),
           "python": platform.python_version(), "numpy": np.__version__, "numba": numba.__version__,
           "triceratops": getattr(triceratops, "__version__", "1.1.0"), "platform": platform.platform(), "processor": platform.processor()}
    print(env, flush=True)
    if args.fast:
        from pipeline import tri_fast
        tri_fast.install(args.cache)
    install()
    runs = []
    for n in args.ns:
        fpp.N_SIM = n
        t = time.perf_counter()
        res = fpp.run_one(d, 1, time.time() + 3600, args.out, args.out / "scratch")
        runs.append(summarise(n, res, time.perf_counter() - t))
        print(json.dumps({k: v for k, v in runs[-1].items() if k != "scenarios"}, indent=1), flush=True)
        out = {"env": env, "tic": d["tic"], "neighbours": res["neighbours_within_10px"], "geometry_sectors": res["geometry_sectors"],
               "aperture": res["aperture"], "n_bins": res["n_bins"], "seed": SEED,
               "setup_seconds": {k: round(v, 2) for k, v in T.setup.items()}, "runs": runs}
        (args.out / f"time_fpp_{args.machine}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
