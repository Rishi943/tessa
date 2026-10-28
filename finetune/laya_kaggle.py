"""Kaggle GPU job: fine-tune Laya (System 1 challenger) on the train split, score val/test/APC.

Bundled next to rows.json (from finetune/laya_rows.py) with requirements.txt `laya==0.4.17`. Writes
/kaggle/working/laya_preds.json: {"base": {toi: P(planet)}, "tuned": {...}}. Platt scaling and the
reject cuts are fitted locally on val, the same way as XGBoost (finetune/system1.py).
"""
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

import laya
from laya.agent import _fix_tokenizer_config
from laya.train import TrainConfig, finetune

HERE = Path(__file__).resolve().parent
TEMP, WORK = Path("/kaggle/temp"), Path("/kaggle/working")
t0 = time.time()


def log(msg):
    print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)


def p_true(answer):
    for key in ("noul", "p_true", "probability"):
        if isinstance(answer.get(key), (int, float)):
            return float(answer[key])
    return float(answer["probabilities"]["true"])


def score(agent, rows, question):
    out = agent.predict_batch([r["state"] for r in rows], question, batch_size=64, max_len=1024)
    return {r["toi"]: p_true(o["answers"]["planet"]) for r, o in zip(rows, out)}


def main():
    data = json.loads((HERE / "rows.json").read_text(encoding="utf-8"))
    question, rows = data["question"], data["rows"]
    base = TEMP / "laya_base"
    snapshot_download("convaiinnovations/laya", local_dir=str(base), allow_patterns=["config.json", "rl_agent_config.json", "model.safetensors", "encoder/*", "tokenizer/*", "*.py"])
    _fix_tokenizer_config(str(base))
    log(f"{len(rows)} rows; GPU {torch.cuda.get_device_name(0)}")

    for split in ("train", "val"):
        with open(TEMP / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == split and "expected" in r:
                    f.write(json.dumps({"state": r["state"], "questions": question, "expected": r["expected"]}) + "\n")
    scored = [r for r in rows if r["split"] != "train"]

    agent = laya.load(str(base), device="cuda")
    log(f"example output {agent.predict_batch([scored[0]['state']], question, max_len=1024)[0]}")
    preds = {"base": score(agent, scored, question)}
    log("base scored")
    del agent
    torch.cuda.empty_cache()

    config = TrainConfig(epochs=4, micro_batch=8, grad_accum=4, max_len=1024, head_max_len=192, seed=3407,
                         eval_data=str(TEMP / "val.jsonl"), log_every=20)
    summary = finetune(data=str(TEMP / "train.jsonl"), model_dir=str(base), output_dir=str(TEMP / "laya_tuned"),
                       config=config, device="cuda")
    log("trained: " + json.dumps({k: v for k, v in summary.items() if isinstance(v, (int, float, str))}))

    agent = laya.load(str(TEMP / "laya_tuned"), device="cuda")
    preds["tuned"] = score(agent, scored, question)
    (WORK / "laya_preds.json").write_text(json.dumps(preds), encoding="utf-8")
    (WORK / "laya_summary.json").write_text(json.dumps(summary, default=str, indent=1), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
