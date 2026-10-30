"""Where TRICERATOPS's time goes: one scenario of each kind at the usual 10^6 draws, profiled by function.

Run with the environment that has TRICERATOPS: python tools/profile_fpp.py
"""
import cProfile
import inspect
import pstats
import time

import numpy as np
import triceratops.marginal_likelihoods as ml

N = 1_000_000
t = np.arange(-0.25, 0.25, 5 / 1440)
flux = 1 - 8e-4 * (np.abs(t) < 0.06)
base = {"time": t, "flux": flux, "sigma": 3e-4, "P_orb": 4.0, "M_s": 1.0, "R_s": 1.0, "Teff": 5700.0, "Z": 0.0,
        "logg": 4.4, "N": N, "parallel": True, "exptime": 600 / 86400, "nsamples": 20, "plx": 5.0,
        "external_lc_files": None, "molusc_file": None, "contrast_curve_file": None, "filt": "TESS", "verbose": 0}


def args(f):
    known = {k: v for k, v in base.items() if k in inspect.signature(f).parameters}
    missing = [k for k, p in inspect.signature(f).parameters.items() if p.default is inspect._empty and k not in known]
    return known, missing


for name in ("lnZ_TTP", "lnZ_TEB", "lnZ_PTP", "lnZ_PEB"):
    f = getattr(ml, name)
    kw, missing = args(f)
    if missing:
        print(name, "needs", missing)
        continue
    np.random.seed(0)
    p = cProfile.Profile()
    t0 = time.time()
    p.enable()
    f(**kw)
    p.disable()
    print(f"\n{name}: {time.time() - t0:.1f} s")
    s = pstats.Stats(p)
    rows = sorted(s.stats.items(), key=lambda x: -x[1][2])[:8]
    for (file, line, func), (_, calls, own, cum, _) in rows:
        print(f"  {own:7.2f} s own {cum:7.2f} s cum {calls:>9} calls  {func} ({file.split(chr(92))[-1]}:{line})")
