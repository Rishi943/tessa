"""Export System 1's exact inputs (the 'clues' feature set) for the scratch-model experiment.

train.npz and val.npz carry labels. test.npz has none: its labels go to sealed/test_labels.npz,
which only evaluate.py --final opens, once, after both models are frozen.

Run from the repo root: python research/scratch-model/data.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from finetune.system1 import binary, load, matrix, y  # noqa: E402
from pipeline.paths import DATA  # noqa: E402
from pipeline.system1 import CLUE_GROUPS  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"


def main() -> None:
    names = json.loads((DATA / "finetune" / "system1" / "results-clues.json").read_text(encoding="utf-8"))["features"]
    rows = load(False, CLUE_GROUPS)
    WORK.mkdir(exist_ok=True)
    (WORK / "sealed").mkdir(exist_ok=True)
    (WORK / "features.json").write_text(json.dumps(names, indent=1), encoding="utf-8")
    for s in ("train", "val", "test"):
        rs = binary([r for r in rows if r["label"]["split"] == s])
        meta = dict(X=matrix(rs, names), tic=np.array([r["tic"] for r in rs]), toi=np.array([str(r["toi"]) for r in rs]),
                    code_reject=np.array([r["code"] == "reject" for r in rs]),
                    agent_bound=np.array([r["code"] == "agent" for r in rs]),
                    km_host=np.array([(r["label"].get("teff") or 9999) < 5300 for r in rs]))
        if s == "test":
            np.savez(WORK / "test.npz", **meta)
            np.savez(WORK / "sealed" / "test_labels.npz", y=y(rs), toi=meta["toi"])
        else:
            np.savez(WORK / f"{s}.npz", y=y(rs), **meta)
        print(s, len(rs), "rows", "" if s == "test" else f"({int(y(rs).sum())} planets)")


if __name__ == "__main__":
    main()
