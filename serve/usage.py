"""Who does Tessa's model work, and what Modal's GPUs were used for.

1. Model calls: data/usage/model_calls.jsonl (agent/usage.js logs every pipeline call) plus the site's
   Ask Tessa calls (D1 table model_calls, via `--d1`), summed by model and purpose: the open-weight share.
2. Modal GPU time: `modal billing report` by hour and app, in GPU-minutes and dollars, written to
   data/usage/modal_gpu.csv. What each run was for is kept in PLAN.md's Modal ledger.

Run: python serve/usage.py [--backfill] [--d1] [--since 2026-10-27]
  --backfill: add calls made before logging existed, from the saved dossiers and traces (once)
"""
import argparse
import csv
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
USAGE = ROOT / "data" / "usage"
CALLS = USAGE / "model_calls.jsonl"
OPEN = ("muse-glimmer",)  # open-weight models, served by us on Modal
GPUS = ("L40S", "H100", "A100", "A10G", "T4", "L4", "B200", "H200")


def backfill() -> int:
    """Calls from before agent/usage.js: one line per saved dossier, investigation and trace."""
    marker = USAGE / ".backfilled"
    if marker.exists():
        print("already backfilled")
        return 0
    rows = []

    def add(path: Path, model: str, purpose: str, tokens: dict | None):
        at = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        rows.append({"at": at, "model": model, "purpose": purpose, "input": (tokens or {}).get("input", 0),
                     "output": (tokens or {}).get("output", 0), "ms": None, "backfilled": True})

    for path in (ROOT / "data" / "dossiers").glob("*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        add(path, d.get("model", path.name.split(".")[1]), "investigation" if ".inv" in path.name else "dossier", d.get("tokens"))
    for folder, model in (("traces", "muse-glimmer"), ("traces_spark", "muse-spark-1.3-contributor")):
        for path in (ROOT / "data" / "finetune" / folder).glob("*.json"):
            d = json.loads(path.read_text(encoding="utf-8"))
            add(path, model, "eval (fine-tune test)" if d.get("split") == "test" else "fine-tune traces", d.get("tokens"))
    USAGE.mkdir(parents=True, exist_ok=True)
    with CALLS.open("a", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["at"]):
            f.write(json.dumps(r) + "\n")
    marker.write_text(datetime.now(timezone.utc).isoformat())
    return len(rows)


def d1_calls() -> list[dict]:
    out = subprocess.run(["npx", "wrangler", "d1", "execute", "tessa", "--remote", "--json", "--command",
                          "SELECT at, model, purpose, input, output FROM model_calls"],
                         cwd=ROOT / "site", capture_output=True, text=True, shell=True, encoding="utf-8")
    try:
        return json.loads(out.stdout)[0]["results"]
    except (json.JSONDecodeError, IndexError, KeyError):
        print("D1: could not read model_calls:", (out.stderr or out.stdout)[-200:])
        return []


def model_table(rows: list[dict]) -> None:
    by = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        k = (r["model"], r["purpose"])
        by[k][0] += 1
        by[k][1] += r.get("input") or 0
        by[k][2] += r.get("output") or 0
    print(f"\n{'model':30} {'purpose':24} {'calls':>7} {'tokens in':>11} {'tokens out':>11}")
    for (m, p), (n, i, o) in sorted(by.items()):
        print(f"{m:30} {p:24} {n:7} {i:11,} {o:11,}")
    tot = lambda keep: [sum(v[j] for (m, _), v in by.items() if keep(m)) for j in range(3)]  # noqa: E731
    o, a = tot(lambda m: m in OPEN), tot(lambda m: True)
    if a[0]:
        print(f"open-weight share: {o[0] / a[0]:.0%} of calls, {(o[1] + o[2]) / max(a[1] + a[2], 1):.0%} of tokens")


def modal_gpu(since: str) -> None:
    out = subprocess.run(["modal", "billing", "report", "--start", since, "-r", "h", "--tz", "local",
                          "--show-resources", "--json"], capture_output=True, text=True, encoding="utf-8")
    try:
        rows = json.loads(out.stdout)
    except json.JSONDecodeError:
        print("Modal billing: no data", out.stderr[-200:])
        return
    rates = {"L40S": 1.95, "H100": 3.95}  # $/GPU-hour, `modal billing rates` 2026-10-28
    gpu = defaultdict(float)
    other = 0.0
    for r in rows:
        if r["resource"] in GPUS:
            gpu[(r["interval_start"][:16], r["description"], r["resource"])] += float(r["cost"])
        else:
            other += float(r["cost"])
    USAGE.mkdir(parents=True, exist_ok=True)
    with (USAGE / "modal_gpu.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hour_start_local", "app", "gpu", "gpu_minutes", "usd"])
        for (h, app, res), usd in sorted(gpu.items()):
            w.writerow([h, app, res, round(usd / rates.get(res, 1) * 60, 1), round(usd, 3)])
    print(f"\n{'hour (local)':17} {'app':16} {'gpu':5} {'GPU-min':>8} {'$':>7}")
    for (h, app, res), usd in sorted(gpu.items()):
        print(f"{h:17} {app:16} {res:5} {usd / rates.get(res, 1) * 60:8.1f} {usd:7.2f}")
    per = defaultdict(float)
    for (_, app, res), usd in gpu.items():
        per[(app, res)] += usd
    for (app, res), usd in sorted(per.items()):
        print(f"total {app:16} {res:5} {usd / rates.get(res, 1):5.2f} GPU-hours  ${usd:.2f}")
    print(f"total GPU ${sum(gpu.values()):.2f} + CPU/memory/network ${other:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--d1", action="store_true", help="include the site's Ask Tessa calls (reads D1 remotely)")
    ap.add_argument("--since", default="2026-10-27")
    args = ap.parse_args()
    if args.backfill:
        print(f"backfilled {backfill()} calls")
    rows = [json.loads(line) for line in CALLS.read_text(encoding="utf-8").splitlines()] if CALLS.exists() else []
    if args.d1:
        rows += d1_calls()
    model_table(rows)
    modal_gpu(args.since)


if __name__ == "__main__":
    main()
