"""Prompt v11: Glimmer starts from System 1's rank and overrules it only with named evidence. Dev only; test is a later, single run.

The same fine-tuned adapter and input as v10, only the system prompt changes. Dev is val + the train signals not in Glimmer's
training rows (finetune/blend.py). The rank line 0.13 is P(planet) 0.6962, set on dev in `python -m finetune.blend scorer`.
Bar, set before any run: at least 118 of 149 planets and at most 11 of 177 FPs on dev.

  python -m finetune.v11 oracle            the v11 rule followed exactly, in code: what obeying the prompt would give
  python -m finetune.v11 run --only-disagree   the 29 signals where v10 and the scorer differ (Modal)
  python -m finetune.v11 run                   every dev signal that reaches the model (Modal; resumes)
  python -m finetune.v11 report            v10 against v11 on what has run
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from finetune.exo_bench import SAVED, tier
from finetune.pilot import investigate, prompt_text, tool_answers, tool_specs, views
from finetune.teacher import load_env, verdict
from finetune.valgate import MODEL, val_signals
from pipeline import gate2
from pipeline.paths import DATA

OUT = DATA / "finetune" / "v11"
RANK = 0.13


def dev_signals() -> list[dict]:
    rows = (DATA / "finetune" / "sft_v10" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    trained = {json.loads(line)["toi"] for line in rows}
    bench = {r["toi"] for r in json.loads((DATA / "finetune" / "exo_bench.json").read_text(encoding="utf-8"))}
    out = []
    for split in ("val", "train"):
        for s in val_signals(split):
            if s["toi"] in bench and s["toi"] not in trained:
                out.append(s | {"split": split, "planet": s["d"]["label"]["target"] == "gate2",
                                "rank": s["d"]["checks"]["system1"]["planet_rank"] or 0})
    return out


def scorer_pass(s: dict) -> bool:
    return s["reaches"] and s["rank"] >= RANK


def hard(s: dict) -> bool:
    return bool(gate2.outside_planet_range(s["d"]["checks"], s["d"].get("evidence"), "v11"))


def oracle() -> None:
    sigs = dev_signals()
    for name, f in (("scorer rule", scorer_pass), ("scorer rule, hard rows and floor stand", lambda s: scorer_pass(s) and not hard(s))):
        print(f"{name}: planets {sum(f(s) for s in sigs if s['planet'])}/{sum(s['planet'] for s in sigs)}, "
              f"FPs {sum(f(s) for s in sigs if not s['planet'])}/{sum(not s['planet'] for s in sigs)}")


def run(only_disagree: bool, jobs: int) -> None:
    load_env()
    OUT.mkdir(exist_ok=True)
    sigs = [s for s in dev_signals() if s["reaches"]]
    if only_disagree:
        sigs = [s for s in sigs if (tier(s, s["split"]) == 2) != scorer_pass(s)]
    print(f"{len(sigs)} signals", flush=True)
    users = views(sigs, "v10", True)
    prompt = prompt_text("v11")
    specs = tool_specs(prompt)

    def work(s):
        f = OUT / f"{s['toi']}.json"
        if f.exists():
            return
        try:
            rec = investigate(prompt, users[s["toi"]], specs, tool_answers(s["toi"]), model=MODEL)
        except Exception as e:
            print(f"{s['toi']}: {str(e)[:200]}", flush=True)
            return
        f.write_text(json.dumps(rec | {"toi": s["toi"], "target": s["d"]["label"]["target"]}), encoding="utf-8")

    with ThreadPoolExecutor(jobs) as pool:
        for n, _ in enumerate(pool.map(work, sigs), 1):
            if n % 25 == 0:
                print(f"{n}/{len(sigs)}", flush=True)


def passes(s: dict) -> bool | None:
    """v11's call on a signal; code-decided signals never pass; None: not run yet."""
    if not s["reaches"]:
        return False
    f = OUT / f"{s['toi']}.json"
    if not f.exists():
        return None
    return verdict(json.loads(f.read_text(encoding="utf-8"))["text"]) == "gate2" and not hard(s)


def report() -> None:
    sigs = dev_signals()
    ran = [s for s in sigs if passes(s) is not None]
    print(f"{len(ran)} of {len(sigs)} dev signals decided ({sum(s['reaches'] for s in ran)} by the model)")
    for name, group in (("planets", [s for s in ran if s["planet"]]), ("FPs", [s for s in ran if not s["planet"]])):
        v10 = sum(tier(s, s["split"]) == 2 for s in group)
        print(f"  {name}: v10 {v10}, v11 {sum(passes(s) for s in group)}, scorer {sum(scorer_pass(s) for s in group)}, of {len(group)}")
    moved = [s for s in ran if s["reaches"] and passes(s) != scorer_pass(s)]
    print(f"  v11 still differs from the scorer on {len(moved)}: "
          f"{sum(s['planet'] != scorer_pass(s) for s in moved)} of them v11 right, {sum(s['planet'] == scorer_pass(s) for s in moved)} wrong")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["oracle", "run", "report"])
    ap.add_argument("--only-disagree", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.stage == "run":
        run(a.only_disagree, a.jobs)
    else:
        {"oracle": oracle, "report": report}[a.stage]()


if __name__ == "__main__":
    main()
