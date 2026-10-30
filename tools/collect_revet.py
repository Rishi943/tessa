"""Summarise the v8 re-vet (data/vet/{survey,batch2}_new) with gate 2. Run when data/vet/*_new.log shows done."""
import json, sys
from pathlib import Path
sys.path.insert(0, "E:/AI/tessa")
from pipeline import gate2
D = Path("E:/AI/tessa/data")
for name in ("survey", "batch2"):
    n = {"reject": 0, "agent": 0}; flag = passed = err = 0
    for p in (D / "vet" / f"{name}_new").glob("*.json"):
        c = json.loads(p.read_text(encoding="utf-8"))
        n[c["code_verdict"]] = n.get(c["code_verdict"], 0) + 1
        if c["code_verdict"] == "agent":
            ep = D / "evidence" / p.name
            ev = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else None
            if gate2.outside_planet_range(c, ev): flag += 1
            else: passed += 1
    print(name, n, "gate2 flag", flag, "pass", passed)
