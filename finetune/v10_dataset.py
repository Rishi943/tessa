"""Training rows for Glimmer on prompt v10, from Spark Open's traces (Built with Muse Spark Open): traces that reach the
true outcome with no invented numbers, as tool calls, tool answers and the final dossier. Reasoning is left out. A
forced trace that corrected a wrong verdict replaces it, without the line that required the tools.
  python -m finetune.v10_dataset                      data/finetune/sft_v10/train.jsonl + val.jsonl
"""
import json
import random
from collections import Counter

from finetune.pilot import FORCE_NOTE, FORCED, SCORE_NOTE, TRACES, invented, prompt_text
from finetune.teacher import verdict
from pipeline.paths import DATA

OUT = DATA / "finetune" / "sft_v10"
HELD = 40  # for the eval loss during training: val and test stay out of the training job


def clean(rec: dict) -> list[dict]:
    rows = []
    for m in rec["messages"]:
        m = {k: v for k, v in m.items() if k != "reasoning"}
        if m["role"] == "system":
            m["content"] = m["content"].replace(FORCE_NOTE, "")
        for call in m.get("tool_calls", []):  # Glimmer's template wants a dict, the API gives a JSON string
            args = call["function"]["arguments"]
            call["function"]["arguments"] = (json.loads(args) if args else {}) if isinstance(args, str) else args
        rows.append(m)
    return rows


def main() -> None:
    prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
    recs = {}
    for f in TRACES.glob("*.json"):
        if f.name != "plan.json":
            r = json.loads(f.read_text(encoding="utf-8"))
            recs[r["toi"]] = r
    for f in FORCED.glob("*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["before"] != r["target"]:
            recs[r["toi"]] = r
    keep = [r for r in recs.values() if verdict(r["text"]) == r["target"] and not invented(r, prompt)]
    random.Random(0).shuffle(keep)
    rows = [{"toi": r["toi"], "target": r["target"], "messages": clean(r)} for r in keep]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in (("val", rows[:HELD]), ("train", rows[HELD:])):
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part), encoding="utf-8")
        print(name, len(part), dict(Counter(r["target"] for r in part)))


if __name__ == "__main__":
    main()
