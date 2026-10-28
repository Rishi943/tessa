"""Rows for Laya, the System 1 challenger: the same check features XGBoost sees, as a small JSON state
(3 significant figures, missing checks left out) with one yes/no question. Train rows are planets vs
false positives with the outcome as the expected answer; val/test/APC rows are scored by the Kaggle job.

Run: python -m finetune.laya_rows OUT_DIR   (writes rows.json for the job folder)
"""
import argparse
import json
from pathlib import Path

from finetune.system1 import load

QUESTION = {"planet": {"type": "noul", "instructions":
                       "Do these vetting checks of a TESS transit signal say it is a real planet, "
                       "not an eclipsing binary, a blended neighbour or noise?"}}


def state(x: dict) -> dict:
    return {k: float(f"{v:.3g}") for k, v in x.items() if v is not None}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    rows = []
    for r in load():
        target = r["label"]["target"]
        rows.append({"toi": r["toi"], "split": r["label"]["split"], "target": target, "state": state(r["x"]),
                     **({"expected": {"planet": target == "gate2"}} if target != "flag" else {})})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rows.json").write_text(json.dumps({"question": QUESTION, "rows": rows}), encoding="utf-8")
    print(f"{len(rows)} rows, {sum(r['split'] == 'train' and 'expected' in r for r in rows)} to train on")


if __name__ == "__main__":
    main()
