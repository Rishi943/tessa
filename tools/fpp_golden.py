"""Reference calls for the fast TRICERATOPS: the exact draws stock TRICERATOPS scores on a real folded signal.

Runs stock lnZ_TTP and lnZ_TEB at 10^6 draws on TIC 283474780's real 418 bins, and saves each likelihood call's
arguments and answers to data/cache/fpp_golden, so a kernel can be compared on real draws in seconds.

Run with the environment that has TRICERATOPS: python tools/fpp_golden.py
"""
import inspect
import json
import time
from pathlib import Path

import numpy as np
import triceratops.marginal_likelihoods as ml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cache" / "fpp_golden"
INPUT = next((ROOT / "data" / "cache" / "fpp_timing_out").rglob("fpp_input_283474780.json"))
stock = {n: getattr(ml, n) for n in ("lnL_TP_p", "lnL_EB_p", "lnL_EB_twin_p")}
calls = []


def recording(name):
    def f(*a, **k):
        names = list(inspect.signature(stock[name]).parameters)
        args = dict(zip(names, a)) | k
        t0 = time.time()
        lnl = stock[name](*a, **k)
        calls.append({"name": name, "seconds": time.time() - t0, "lnL": lnl, **args})
        print(f"{name}: {len(lnl)} draws in {calls[-1]['seconds']:.1f} s", flush=True)
        return lnl
    return f


def main() -> None:
    d = json.loads(INPUT.read_text(encoding="utf-8"))
    t, flux, sigma = np.array(d["time"]), np.array(d["flux"]), float(d["flux_err"])
    print(f"{len(t)} bins, sigma {sigma:.2e}, period {d['period']:.4f} d, depth {d['depth']:.2e}")
    base = dict(time=t, flux=flux, sigma=sigma, P_orb=float(d["period"]), M_s=1.0, R_s=1.0, Teff=5700.0, Z=0.0,
                N=1_000_000, parallel=True, exptime=d["exptime_s"] / 86400, nsamples=20)
    for name in stock:
        setattr(ml, name, recording(name))
    OUT.mkdir(parents=True, exist_ok=True)
    for scenario in ("lnZ_TTP", "lnZ_TEB"):
        np.random.seed(0)
        f = getattr(ml, scenario)
        res = f(**{k: v for k, v in base.items() if k in inspect.signature(f).parameters})
        res = res if isinstance(res, tuple) else (res,)
        print(scenario, "lnZ", [float(r["lnZ"]) for r in res])
        np.savez_compressed(OUT / f"{scenario}.npz", lnZ=np.array([float(r["lnZ"]) for r in res]))
    for i, c in enumerate(calls):
        np.savez_compressed(OUT / f"{i}_{c['name']}.npz", **{k: np.asarray(v) for k, v in c.items()})


if __name__ == "__main__":
    main()
