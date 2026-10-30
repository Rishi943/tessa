"""Which re-vetted stars need rerunning under vetting v9 (pipeline/vet.py RULE_REJECT_RANK).

The v8 re-vet (Kaggle tessa-revet-0/1/2) rejected a star outright when the phase-curve rule fired ("the star varies by"),
before System 1, the centroid and the tools ran. v9 lets System 1 score it first. Two kinds need the star vetted again:
  A  rejected with a "the star varies" reason, no System 1 score, and no known-object or duration reason (those stay hard)
  B  scored by System 1 but rejected only by the centroid or phase-curve rules; v9's verdict (from the stored checks) is
     agent, and the tools' results (checks.tools, data/evidence/TIC.json) were never made
Everything else keeps its v8 verdict under v9.

Run: python tools/revet_v9.py [VET_DIR ...] [--out tics.txt]   (default: the downloaded Kaggle outputs and data/vet/*_new)
The comma-separated TICs go to `python -m pipeline.revet ... --tics` (or into a pipeline.revet export without --skip).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.vet import verdict  # noqa: E402

VET = Path(__file__).resolve().parent.parent / "data" / "vet"


def default_dirs() -> list[Path]:
    out = [d for d in VET.glob("kaggle_revet_*/notebooks/*/tessa-revet-*/vet/*") if d.is_dir()]
    return out + [VET / "survey_new", VET / "batch2_new"]


def kind(checks: dict) -> str | None:
    if checks["code_verdict"] != "reject":
        return None
    reasons = checks["reasons"]
    if checks.get("system1") is None:
        varies = any(r.startswith("the star varies") for r in reasons)
        hard = any(r.startswith(("known:", "transit ")) for r in reasons)
        return "A" if varies and not hard else None
    return "B" if verdict(checks)[0] == "agent" else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vet_dirs", nargs="*", type=Path)
    ap.add_argument("--out", type=Path, help="write the TICs, comma separated")
    args = ap.parse_args()
    found, seen = {"A": [], "B": []}, set()
    dirs = args.vet_dirs or default_dirs()
    for d in dirs:
        for path in sorted(d.glob("[0-9]*.json")):
            checks = json.loads(path.read_text(encoding="utf-8"))
            if checks["tic"] in seen:
                continue
            seen.add(checks["tic"])
            if k := kind(checks):
                found[k].append(checks["tic"])
    print(f"{len(seen)} stars in {len(dirs)} folders; rerun A (no System 1 score, rejected by the phase-curve rule) "
          f"{len(found['A'])}, B (scored, v9 verdict agent, no tools) {len(found['B'])}")
    if args.out:
        args.out.write_text(",".join(map(str, sorted(found["A"] + found["B"]))), encoding="utf-8")


if __name__ == "__main__":
    main()
