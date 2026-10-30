"""A whole fast TRICERATOPS run on one signal, profiled function by function: what's left to speed up.

Run: E:/AI/_ref/tri-venv/Scripts/python tools/profile_run.py [TIC]
"""
import cProfile
import json
import pstats
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import fpp, tri_fast  # noqa: E402

tic = sys.argv[1] if len(sys.argv) > 1 else "283474780"
d = json.loads(next((ROOT / "data" / "cache" / "fpp_timing_out").rglob(f"fpp_input_{tic}.json")).read_text(encoding="utf-8"))
out = ROOT / "data" / "cache" / "fpp_profile"
out.mkdir(parents=True, exist_ok=True)
fpp.retrying_downloads()
tri_fast.install()
tri_fast.lnL_TP_p(np.zeros(4), np.ones(4), 1.0, np.ones(2), np.full(2, 4.0), np.full(2, 89.0), np.full(2, 1e12), np.ones(2),
                  np.full(2, 0.3), np.full(2, 0.2), np.zeros(2), np.zeros(2), np.zeros(2))  # compile outside the profile
np.random.seed(0)
p = cProfile.Profile()
t0 = time.time()
p.enable()
res = fpp.run_one(d, 1, time.time() + 3600, out, out / "scratch")
p.disable()
print(f"total {time.time() - t0:.0f} s, FPP {res.get('FPP_mean')}, NFPP {res.get('NFPP_mean')}")
s = pstats.Stats(p)
rows = [(cum, own, calls, f"{func} ({Path(file).name}:{line})") for (file, line, func), (_, calls, own, cum, _) in s.stats.items()]
print("\nby time inside the function")
for cum, own, calls, name in sorted(rows, key=lambda r: -r[1])[:22]:
    print(f"  {own:7.2f} own {cum:7.2f} cum {calls:>8}  {name}")
print("\nby cumulative time, TRICERATOPS's and our own")
keep = ("triceratops", "funcs", "priors", "marginal", "fpp.py", "tri_fast", "query", "download", "calc_", "lnZ", "Tesscut", "get_")
for cum, own, calls, name in sorted((r for r in rows if any(k in r[3] for k in keep)), key=lambda r: -r[0])[:30]:
    print(f"  {own:7.2f} own {cum:7.2f} cum {calls:>8}  {name}")
