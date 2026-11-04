"""Gate 2 again, under the rule as it now stands, for every signal flagged only for having no test window, or for --tics
(pipeline/gate2.py: a year's windows are listed, and the season test decides). Updates each batch's status.csv row
and candidates/ file the way pipeline.gate2 writes them, so pipeline.export picks the result up.

  python tools/regate_notest.py [--jobs 3]
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.paths import DATA, batches  # noqa: E402


def one(job: tuple[str, int]) -> tuple[str, int, str, list, dict]:
    from pipeline.gate2 import decide
    from pipeline.vet import signal_from_results
    batch, tic = job
    results, lc = batches()[batch]
    checks = json.loads((DATA / "vet" / f"{batch}_v10" / f"{tic}.json").read_text(encoding="utf-8"))
    dossier = json.loads((DATA / "dossiers" / f"{tic}.glimmer-vetter-v10.v10.json").read_text(encoding="utf-8"))
    status, reasons, extra = decide(checks, dossier, signal_from_results(tic, results), lc)
    return batch, tic, status, reasons, extra if status == "candidate" else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--tics", help="comma-separated: rerun these instead (e.g. after Glimmer answered them again)")
    a = ap.parse_args()
    want = {int(t) for t in a.tics.split(",")} if a.tics else None
    jobs = []
    for batch in batches():
        p = DATA / "vet" / f"{batch}_v10" / "status.csv"
        if p.exists():
            jobs += [(batch, int(r["tic"])) for r in csv.DictReader(p.open(encoding="utf-8"))
                     if (int(r["tic"]) in want if want else
                         r["status"] == "flagged" and r["reasons"].startswith("no test"))]
    print(f"{len(jobs)} signals", flush=True)
    done = {}
    with ProcessPoolExecutor(a.jobs) as pool:
        for batch, tic, status, reasons, extra in pool.map(one, jobs):
            done[(batch, tic)] = (status, reasons)
            print(f"TIC {tic} ({batch}): {status}" + (f" ({reasons[0][:110]})" if reasons else ""), flush=True)
            vet = DATA / "vet" / f"{batch}_v10"
            if status == "candidate":
                checks = json.loads((vet / f"{tic}.json").read_text(encoding="utf-8"))
                (vet / "candidates").mkdir(exist_ok=True)
                (vet / "candidates" / f"{tic}.json").write_text(json.dumps(
                    {"tic": tic, "signal": checks["signal"], "size": checks["size"], "star": checks["star"], **extra},
                    indent=1, default=float), encoding="utf-8")
    for batch in {b for b, _ in done}:
        p = DATA / "vet" / f"{batch}_v10" / "status.csv"
        rows = list(csv.DictReader(p.open(encoding="utf-8")))
        for r in rows:
            if (batch, int(r["tic"])) in done:
                r["status"], reasons = done[(batch, int(r["tic"]))]
                r["reasons"] = "; ".join(reasons)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, ["tic", "status", "reasons"])
            w.writeheader()
            w.writerows(rows)
    from collections import Counter
    print(Counter(s for s, _ in done.values()))


if __name__ == "__main__":
    main()
