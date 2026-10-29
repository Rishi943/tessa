"""A benchmark of Claude as the vetter, on a stratified sample of the agent-bound test TOIs (prompt v6).

Claude's dossiers are a comparison row only: Anthropic's terms bar training other models on Claude's outputs,
so none of them go into Glimmer's training data.

  python -m finetune.claude_bench sample [--n-planet 50 --n-fp 35 --n-apc 15]
      writes data/finetune/claude_bench/{system.md, inputs.jsonl, key.json}: the prompt, the inputs without
      their outcomes (what the agent reads) and the outcomes (what it must never read)
  python -m finetune.claude_bench score NAME=FOLDER [...]
      every model's verdicts on the same sample: FOLDER holds TOI.json files with a "verdict" (Claude's) or
      TOI.0.json trace files (the Glimmer and Spark runs)
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from pipeline.paths import DATA

FOLDER = DATA / "finetune"
OUT = FOLDER / "claude_bench"


def sample(n_planet: int, n_fp: int, n_apc: int) -> None:
    rows = [json.loads(line) for line in (FOLDER / "prompts_v6_test.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(20261029)
    picked = []
    for target, n in (("gate2", n_planet), ("reject", n_fp), ("flag", n_apc)):
        pool = [r for r in rows if r["target"] == target]
        picked += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(picked)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "system.md").write_text((Path(__file__).parent.parent / "agent" / "prompts" / "vetting.v6.md").read_text(encoding="utf-8"), encoding="utf-8")
    with (OUT / "inputs.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(picked):
            f.write(json.dumps({"id": f"s{i:03d}", "user": r["user"]}) + "\n")
    (OUT / "key.json").write_text(json.dumps({f"s{i:03d}": {"toi": r["toi"], "target": r["target"]} for i, r in enumerate(picked)}, indent=1), encoding="utf-8")
    print(f"{len(picked)} signals: {dict(Counter(r['target'] for r in picked))}")


def verdicts(folder: Path, key: dict) -> dict:
    by_toi = {v["toi"]: k for k, v in key.items()}
    out = {}
    for path in folder.glob("*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        sid = path.stem if path.stem in key else by_toi.get(d.get("toi")) if path.stem.endswith(".0") else None
        if sid:
            out[sid] = d.get("verdict") or (d.get("dossier") or {}).get("verdict")
    return out


def score(models: list[str]) -> None:
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))
    for spec in models:
        name, folder = spec.split("=", 1)
        v = verdicts(Path(folder), key)
        c = Counter((key[s]["target"], v.get(s, "missing")) for s in key)
        planets, fps = sum(t == "gate2" for t in (x["target"] for x in key.values())), sum(t == "reject" for t in (x["target"] for x in key.values()))
        lost = c[("gate2", "reject")]
        caught = c[("reject", "reject")]
        print(f"{name:>14}: {len(v)} of {len(key)} | planets lost {lost}/{planets}, passed {c[('gate2', 'gate2')]}, "
              f"FPs caught {caught}/{fps}, passed as planets {c[('reject', 'gate2')]} | ambiguous flagged {c[('flag', 'flag')]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n-planet", type=int, default=50)
    s.add_argument("--n-fp", type=int, default=35)
    s.add_argument("--n-apc", type=int, default=15)
    sc = sub.add_parser("score")
    sc.add_argument("models", nargs="+")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.n_planet, a.n_fp, a.n_apc)
    else:
        score(a.models)


if __name__ == "__main__":
    main()
