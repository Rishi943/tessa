"""The labelled TOIs as the pipeline with System 1 hands them to the vetter (prompt v6).

The fine-tune vet files were made with the old rules (odd/even, secondary and size as vetoes), so which TOIs
reach the agent is redone here the way pipeline/vet.verdict does it with System 1: the junk rules stay hard,
the centroid fills in where the rules had skipped it, and the signal goes on only if System 1 passes it and
the dip isn't on a Gaia star. Each file gets System 1's score and drivers (out-of-fold for train, from
finetune/system1.py's s1-clues.json) and the 5 most similar train TOIs with what follow-up found.

Run: python -m finetune.s1_inputs   (after finetune/system1.py --name clues --clues all and finetune/similar.py)
Writes data/finetune/vet_s1/TOI.json and prints the agent-bound counts by split and outcome.
"""
import json
from collections import Counter

import numpy as np

from pipeline.paths import DATA
from pipeline.system1 import LABELS, NAME, model
from pipeline.vet import junk, verdict

FOLDER = DATA / "finetune"
OUT = FOLDER / "vet_s1"


def main() -> None:
    m = model()
    s1 = json.loads((FOLDER / "system1" / f"s1-{NAME}.json").read_text(encoding="utf-8"))
    similar = json.loads((FOLDER / "similar.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    counts = Counter()
    for path in sorted((FOLDER / "vet").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["toi"] not in s1:
            continue
        c = d["checks"]
        clue_path = FOLDER / "clues" / f"{d['toi']}.json"
        clue = json.loads(clue_path.read_text(encoding="utf-8")) if clue_path.exists() else {}
        if not c.get("centroid") and clue.get("centroid") and not junk(c):
            c["centroid"] = clue["centroid"]  # with System 1 the centroid runs on every signal past the junk rules
        p = s1[d["toi"]]["p_planet"]
        c["system1"] = {"p_planet": p, "cut": m["cut"], "passes": p >= m["cut"],
                        "planet_rank": round(float(np.searchsorted(m["planets"], p) / len(m["planets"])), 3),
                        "drivers": [{"check": n, "label": LABELS.get(n, n), "value": v, "log_odds": lo}
                                    for n, v, lo in s1[d["toi"]]["drivers"]],
                        "model": f"xgb-{NAME}"}
        c["similar"] = [{"target": s["target"], "reason": s["reason"]} for s in similar.get(d["toi"], [])]
        c["code_verdict"], c["reasons"] = verdict(c)
        (OUT / path.name).write_text(json.dumps(d), encoding="utf-8")
        counts[(d["label"]["split"], d["label"]["target"], c["code_verdict"])] += 1
    for split in ("train", "val", "test"):
        row = {t: f"{counts[(split, t, 'agent')]}/{counts[(split, t, 'agent')] + counts[(split, t, 'reject')]}"
               for t in ("gate2", "reject", "flag")}
        print(f"{split}: to the agent (of all) {row}")


if __name__ == "__main__":
    main()
