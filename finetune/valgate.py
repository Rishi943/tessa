"""The v10 val gate (#42): does the fine-tuned Glimmer, deciding with tools and XGBoost's rank as evidence,
pass enough planets and catch enough false positives on the held-out validation signals?

By default only val is read; `--split test` runs the same frozen setup once on test (the saved verdicts that
finetune/exo_bench.py and tools/glimmer_value.py report). For every val signal whose follow-up settled it a planet (gate2) or not (reject):
code's hard junk and rule rejects stand as before; everything else goes to the fine-tuned Glimmer (served on
Modal, model glimmer-vetter-v10) with the v10 prompt, the XGBoost rank, and the same cached tool answers the
training used. A signal ends a candidate when Glimmer says gate2 and no HARD_ROWS row fires.

Bar (set before the run): at least 75 of 110 planets passed, at most 3 of 106 FPs through.
  python -m finetune.valgate [--smoke] [--jobs 6]
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pipeline import gate2
from pipeline.paths import DATA

from finetune.pilot import (SCORE_NOTE, VET, candidates, investigate, invented, parse, prompt_text,
                            system_one_only, tool_answers, tool_specs, views)
from finetune.teacher import load_env, verdict

MODEL = "glimmer-vetter"  # the fine-tuned v10 adapter on Modal
OUT = DATA / "finetune" / "valgate"


def val_signals(split: str = "val") -> list[dict]:
    """Signals of the split with target gate2 or reject, each with whether code decides it (junk/rule reject that
    the model never sees) or it reaches the model (agent-bound, or a reject only the System 1 cut made)."""
    out = []
    for f in sorted(VET.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["label"]["split"] != split or d["label"]["target"] not in ("gate2", "reject"):
            continue
        c = d["checks"]
        s1_only = c["code_verdict"] == "reject" and system_one_only(c["reasons"])
        reaches = c["code_verdict"] == "agent" or s1_only
        out.append({"toi": d["toi"], "d": d, "s1_only": s1_only, "reaches": reaches})
    return out


def run(smoke: bool, jobs: int, split: str = "val", model: str = MODEL) -> None:
    load_env()
    out = OUT if split == "val" else OUT.with_name(f"{split}gate")
    if model != MODEL:  # another model on the same signals keeps its own folder: testgate_glimmer-base
        out = out.with_name(f"{out.name}_{model}")
    globals()["OUT"] = out
    out.mkdir(parents=True, exist_ok=True)
    sigs = val_signals(split)
    model_sigs = [s for s in sigs if s["reaches"]]
    # tool coverage for the signals the model will see
    missing = [s["toi"] for s in model_sigs if not any(DATA.joinpath("finetune", "teacher").rglob(f"tools_{s['toi']}.json"))]
    print(f"val: {sum(s['d']['label']['target'] == 'gate2' for s in sigs)} planets, "
          f"{sum(s['d']['label']['target'] == 'reject' for s in sigs)} FPs; "
          f"{len(model_sigs)} reach the model ({len(sigs) - len(model_sigs)} decided by code); "
          f"tool answers missing for {len(missing)}", flush=True)
    if missing:
        print("  missing:", missing[:10], flush=True)
        model_sigs = [s for s in model_sigs if s["toi"] not in missing]
    if smoke:
        pl = [s for s in model_sigs if s["d"]["label"]["target"] == "gate2"][:2]
        fp = [s for s in model_sigs if s["d"]["label"]["target"] == "reject"][:2]
        model_sigs = pl + fp
        print(f"smoke: {len(model_sigs)} signals (cold start warms the GPU, allow minutes)", flush=True)

    users = views(model_sigs, "v10", True)
    prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
    specs = tool_specs(prompt)

    def work(s):
        saved = OUT / f"{s['toi']}.json"
        if saved.exists():  # resume: don't pay for a signal already run
            return json.loads(saved.read_text(encoding="utf-8"))
        answers = tool_answers(s["toi"])
        try:
            rec = investigate(prompt, users[s["toi"]], specs, answers, model=model)
        except Exception as e:
            print(f"{s['toi']}: {str(e)[:200]}", flush=True)
            return None
        rec |= {"toi": s["toi"], "target": s["d"]["label"]["target"], "user": users[s["toi"]],
                "answers": {k: answers[k] for k in set(rec["tools"]) if k in answers}}
        (OUT / f"{s['toi']}.json").write_text(json.dumps(rec), encoding="utf-8")
        return rec

    with ThreadPoolExecutor(jobs) as pool:
        recs = [r for r in pool.map(work, model_sigs) if r]

    by_toi = {r["toi"]: r for r in recs}
    planets = passed = fps = through = 0
    conf = Counter()
    inv = 0
    for s in sigs:
        d, tgt = s["d"], s["d"]["label"]["target"]
        hard = gate2.outside_planet_range(d["checks"], d.get("evidence"), "v10")
        if s["reaches"] and s["toi"] in by_toi:
            v = verdict(by_toi[s["toi"]]["text"])
            conf[(tgt, v)] += 1
            inv += bool(invented(by_toi[s["toi"]], prompt))
            is_candidate = v == "gate2" and not hard  # hard now includes the v10 rank floor
        elif s["reaches"]:
            continue  # ran the model but it errored or had no tools: leave out of the tally
        else:
            is_candidate = False  # code rejected it
            conf[(tgt, "code-reject")] += 1
        if tgt == "gate2":
            planets += 1
            passed += is_candidate
        else:
            fps += 1
            through += is_candidate

    rank = Counter((s["d"]["label"]["target"], (s["d"]["checks"]["system1"]["planet_rank"] or 0) >= gate2.PASS_RANK) for s in sigs)
    lines = [f"\n=== v10 {split} gate ===",
             f"planets passed (recovered): {passed}/{planets}   ({passed / planets:.0%})",
             f"FPs through (leaked):       {through}/{fps}   ({through / fps:.1%})",
             f"model verdicts (target -> verdict): " + ", ".join(f"{t}->{v}={n}" for (t, v), n in sorted(conf.items(), key=lambda x: (x[0][0], str(x[0][1])))),
             f"invented numbers in {inv} model dossiers",
             f"XGBoost rank>=0.32 baseline: planets {rank[('gate2', True)]}/{planets} pass, FPs {rank[('reject', True)]}/{fps} through",
             f"(ran {len(recs)} model signals)"]
    text = "\n".join(lines)
    print(text, flush=True)
    (OUT / "summary.txt").write_text(text + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="4 signals, to warm the GPU and check the loop")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--model", default=MODEL, help="glimmer-base: the same server without the adapter")
    a = ap.parse_args()
    run(a.smoke, a.jobs, a.split, a.model)


if __name__ == "__main__":
    main()
