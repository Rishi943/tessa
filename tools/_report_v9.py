import json, sys
from pathlib import Path
sys.path.insert(0, "E:/AI/tessa")
from pipeline import gate2, vet
D = Path("E:/AI/tessa/data")
for name in ("survey", "batch2"):
    n = {"reject": 0, "pass": 0, "flag": 0}; look = []
    files = list((D / "vet" / f"{name}_new").glob("[0-9]*.json"))
    for p in files:
        c = json.loads(p.read_text(encoding="utf-8"))
        if c["code_verdict"] == "reject": n["reject"] += 1; continue
        ev = None
        for ed in ("evidence_new", "evidence"):
            e = D / ed / p.name
            if e.exists(): ev = json.loads(e.read_text(encoding="utf-8")); break
        if gate2.outside_planet_range(c, ev): n["flag"] += 1
        else:
            n["pass"] += 1
            if vet.on_gaia_star(c) or vet.star_varies(c): look.append(p.stem)
    print(name, len(files), n, "look:", look)
for t in (50099421, 61220826):
    for name in ("survey", "batch2"):
        p = D / "vet" / f"{name}_new" / f"{t}.json"
        if p.exists():
            c = json.loads(p.read_text(encoding="utf-8")); print(t, name, c["code_verdict"], c["reasons"][:2])
