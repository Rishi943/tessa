"""SFT data for the Glimmer vetter, from base Glimmer's own dossiers on the training TOIs (rejection sampling).

A trace is kept when its verdict matches what follow-up found, it cites no invented numbers and it finished
(not cut off at the token limit); at most two per TOI, one low and one high reasoning strength when both
exist. Hinted traces (written with the outcome in the prompt, see traces.js --hint) fill in TOIs that had
none, up to HINT_SHARE of the set, and only when their reasoning doesn't mention the hint. The prompt each
example trains on is the plain one, never the hinted one.

Run: python -m finetune.dataset [--traces traces_v6 --out sft_v6]
Writes data/finetune/sft/{train,val}.jsonl and prints what was kept.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict

from pipeline.paths import DATA

FOLDER = DATA / "finetune"
PER_TOI = 2
HINT_SHARE = 0.25
LEAK = re.compile(r"follow-up|follow up|hint|was told|we know|known outcome|outcome|TFOP|confirmed planet|"
                  r"false positive was|disposition", re.IGNORECASE)


def good(t: dict, planet_flag: bool = False) -> bool:
    """A trace worth training on. planet_flag: for a planet, "flag" counts too. Gate 2's own ranges flag most
    real planets (bright, deep transits vary between sectors at high significance), so the prompt asks for a flag
    there; the costly mistakes are rejecting a planet and passing a false positive."""
    right = t.get("verdict") == t["target"] or (planet_flag and t["target"] == "gate2" and t.get("verdict") == "flag")
    return right and not t.get("invented") and t.get("stop") == "stop" and t.get("thinking")


def example(t: dict) -> dict:
    return {"toi": t["toi"], "target": t["target"], "verdict": t["verdict"], "effort": t["effort"], "hinted": bool(t.get("hint")),
            "messages": [{"role": "system", "content": t["system"]}, {"role": "user", "content": t["user"]},
                         {"role": "assistant", "reasoning_content": t["thinking"], "content": t["text"]}]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", default="traces", help="folder in data/finetune with the trace files (v6: traces_v6)")
    ap.add_argument("--out", default="sft", help="folder in data/finetune for train.jsonl and val.jsonl")
    ap.add_argument("--planet-flag", action="store_true", help="a planet's trace may say flag (v6)")
    ap.add_argument("--per-target", default="", help="traces kept per TOI by outcome, e.g. gate2=1,flag=1,reject=2")
    ap.add_argument("--repeat", default="", help="repeat an outcome's examples, e.g. reject=3, so the rare verdict is not drowned out")
    args = ap.parse_args()
    per = {k: int(v) for k, v in (x.split("=") for x in args.per_target.split(",") if x)}
    repeat = {k: int(v) for k, v in (x.split("=") for x in args.repeat.split(",") if x)}
    by_toi = defaultdict(list)
    for path in (FOLDER / args.traces).glob("*.json"):
        t = json.loads(path.read_text(encoding="utf-8"))
        if t["split"] in ("train", "val"):
            by_toi[t["toi"]].append(t)
    rng = random.Random(20261028)
    out = {"train": [], "val": []}
    stats = Counter()
    for toi, traces in sorted(by_toi.items()):
        split, target = traces[0]["split"], traces[0]["target"]
        plain = [t for t in traces if not t.get("hint") and good(t, args.planet_flag)]
        hinted = [t for t in traces if t.get("hint") and good(t, args.planet_flag) and not LEAK.search(t["thinking"])]
        # the outcome itself first (a planet's gate2 before its flag), one low and one high strength when both exist
        plain.sort(key=lambda t: t["verdict"] != target)
        picked = []
        for effort in ("low", "high"):
            picked += [t for t in plain if t["effort"] == effort][:1]
        picked = sorted(picked + [t for t in plain if t not in picked], key=lambda t: t["verdict"] != target)[:per.get(target, PER_TOI)]
        stats[f"{split} TOIs"] += 1
        if picked:
            stats[f"{split} with a correct trace"] += 1
            out[split] += [example(t) for t in picked] * repeat.get(target, 1)
        elif hinted:
            out[split] += [example(hinted[0])] * repeat.get(target, 1)
            stats[f"{split} hinted"] += 1
    for split, rows in out.items():
        hinted = [r for r in rows if r["hinted"]]
        cap = int(HINT_SHARE * len(rows))
        if len(hinted) > cap:  # keep the plain ones, drop hinted ones at random down to the cap
            drop = {id(r) for r in rng.sample(hinted, len(hinted) - cap)}
            rows[:] = [r for r in rows if id(r) not in drop]
        rng.shuffle(rows)
        (FOLDER / args.out).mkdir(exist_ok=True)
        with (FOLDER / args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        mix = Counter(f"{r['target']}->{r['verdict']}" for r in rows)
        print(f"{split}: {len(rows)} examples {dict(mix)}, efforts {dict(Counter(r['effort'] for r in rows))}, "
              f"hinted {sum(r['hinted'] for r in rows)}")
    print(dict(stats))


if __name__ == "__main__":
    main()
