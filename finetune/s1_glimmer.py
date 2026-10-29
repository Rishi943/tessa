# System 1 as Muse Glimmer: the same checks XGBoost reads (production clues + per-sector odd/even), as a small
# JSON state, with one yes/no question, QLoRA-tuned on the train split's planets and false positives. Its
# P(planet) is P("yes") against P("no") for the answer. A Meta model in System 1's place, measured on val against
# XGBoost like any System 1 change; the test split is scored and saved but not read until the final run.
#   python -m finetune.s1_glimmer rows [pct]                     # data/finetune/s1_glimmer/rows[-pct].json
#   modal run --detach finetune/s1_glimmer.py --dry              # 5 steps: the rendered example and the mask
#   modal run --detach finetune/s1_glimmer.py [--encoding pct --epochs 3]   # train + score; tessa-ft glimmer-s1[-pct]/
#   modal volume get tessa-ft glimmer-s1/preds.json data/finetune/s1_glimmer/   (glimmer-s1-pct: preds-pct.json)
#   python -m finetune.s1_glimmer compare [pct ...]              # val: AUC and FPs caught at 1% / 2% / 5% planets lost
# pct: each check as its percentile among the train split's planets (language models read ranks better than raw
# numbers: TabLLM-style serialisation)
import json
import sys
from pathlib import Path

MAX_SEQ = 2048
SYSTEM = ("You are System 1 for Tessa, a planet search in NASA TESS light curves. Given the vetting checks of one "
          "transit signal, answer yes if it is a real planet and no if it is not (an eclipsing binary, a blended "
          "neighbour or noise). Answer with one word: yes or no.")
SYSTEM_PCT = ("You are System 1 for Tessa, a planet search in NASA TESS light curves. Given the vetting checks of one "
              "transit signal, each as its percentile among confirmed planets (50: as typical as the median planet; "
              "0 or 100: lower or higher than every planet), answer yes if it is a real planet and no if it is not "
              "(an eclipsing binary, a blended neighbour or noise). Answer with one word: yes or no.")
FOLDER = Path("data/finetune/s1_glimmer")
GPUS = ["H100", "A100-80GB", "L40S"]  # whichever is free: the 4-bit base peaks near 25 GiB


def messages(state: dict, answer: str | None = None, system: str = SYSTEM) -> list[dict]:
    m = [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(state, separators=(",", ":"))}]
    return m + ([{"role": "assistant", "content": answer}] if answer else [])


if __name__ != "__main__":  # Modal's side; the local commands below need no modal package
    import modal

    from finetune.train import BASE, image, out, weights

    image = image.add_local_python_source("finetune")  # the container imports finetune.train too

    app = modal.App("tessa-s1-glimmer")


    @app.function(image=image, gpu=GPUS, volumes={"/root/.cache/huggingface": weights, "/out": out}, timeout=4 * 60 * 60)
    def run(rows: list[dict], epochs: float = 2.0, rank: int = 16, lr: float = 2e-4, dry: bool = False,
            encoding: str = "raw"):
        system = SYSTEM_PCT if encoding == "pct" else SYSTEM
        from unsloth import FastModel  # first, so it patches transformers
        from unsloth.chat_templates import train_on_responses_only
        import torch
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        model, tokenizer = FastModel.from_pretrained(model_name=BASE, max_seq_length=MAX_SEQ, load_in_4bit=True)
        model = FastModel.get_peft_model(
            model, finetune_vision_layers=False, finetune_language_layers=True,
            finetune_attention_modules=True, finetune_mlp_modules=True,
            r=rank, lora_alpha=rank, lora_dropout=0, bias="none",
            use_gradient_checkpointing="unsloth", random_state=3407,
        )

        def render(state, answer):
            return tokenizer.apply_chat_template(messages(state, answer, system), tokenize=False)                 .removeprefix(tokenizer.bos_token)

        train = [r for r in rows if r["split"] == "train" and r["target"] in ("gate2", "reject")]
        ds = Dataset.from_list([{"text": render(r["state"], "yes" if r["target"] == "gate2" else "no")} for r in train])
        print(f"train {len(ds)} ({sum(r['target'] == 'gate2' for r in train)} planets); example:\n{ds[0]['text']}", flush=True)
        trainer = SFTTrainer(
            model=model, tokenizer=tokenizer, train_dataset=ds,
            args=SFTConfig(
                dataset_text_field="text", max_length=MAX_SEQ, packing=False,
                per_device_train_batch_size=8, gradient_accumulation_steps=2,  # effective batch 16
                num_train_epochs=epochs, max_steps=5 if dry else -1,
                learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
                optim="adamw_8bit", seed=3407, bf16=True, logging_steps=10, save_strategy="no",
                output_dir="/out/s1-checkpoints", report_to="none",
            ),
        )
        trainer = train_on_responses_only(trainer, instruction_part="<|start|>user<|message|>",
                                          response_part="<|start|>assistant")
        labels = trainer.train_dataset[0]["labels"]
        first = next(i for i, x in enumerate(labels) if x != -100)
        print("trained part:", repr(tokenizer.decode(trainer.train_dataset[0]["input_ids"][first:])), flush=True)
        trainer.train()
        name = "glimmer-s1" + ("" if encoding == "raw" else f"-{encoding}") + ("-dry" if dry else "")
        model.save_pretrained(f"/out/{name}")
        tokenizer.save_pretrained(f"/out/{name}")

        # P(yes) for every row outside train: the two answers' log-likelihoods after the same prompt
        FastModel.for_inference(model)
        score_rows = [r for r in rows if r["split"] != "train"][: 8 if dry else None]
        preds = {}
        with torch.no_grad():
            for n, r in enumerate(score_rows, 1):
                ll = []
                for answer in ("yes", "no"):
                    ids = tokenizer(text=[render(r["state"], answer)], return_tensors="pt")["input_ids"].to(model.device)
                    prompt = tokenizer(text=[render(r["state"], "yes" if answer == "no" else "no")])["input_ids"][0]
                    k = next(i for i, (a, b) in enumerate(zip(ids[0].tolist(), prompt)) if a != b)  # first answer token
                    logp = torch.log_softmax(model(input_ids=ids).logits[0, k - 1:-1].float(), -1)
                    ll.append(float(logp.gather(1, ids[0, k:].unsqueeze(1)).sum()))
                preds[r["toi"]] = float(torch.sigmoid(torch.tensor(ll[0] - ll[1])))
                if n % 100 == 0:
                    print(f"scored {n}/{len(score_rows)}", flush=True)
        Path(f"/out/{name}/preds.json").write_text(json.dumps(preds))
        out.commit()
        print(f"done: {len(preds)} scored; peak memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)


    @app.local_entrypoint()
    def main(dry: bool = False, epochs: float = 2.0, encoding: str = "raw"):
        rows = json.loads((FOLDER / ("rows.json" if encoding == "raw" else f"rows-{encoding}.json")).read_text(encoding="utf-8"))
        call = run.spawn(rows, epochs=epochs, dry=dry, encoding=encoding)
        print(f"spawned {call.object_id}; preds in tessa-ft glimmer-s1{'' if encoding == 'raw' else '-' + encoding}/preds.json")


def make_rows(encoding: str = "raw") -> None:
    import numpy as np
    from finetune.laya_rows import state
    from finetune.system1 import CLUE_GROUPS, load

    rows = load(groups=CLUE_GROUPS)  # the production clues, the sector checks among them
    if encoding == "pct":  # each check's percentile among the train planets, so no val or test label leaks in
        planets = [r["x"] for r in rows if r["label"]["split"] == "train" and r["label"]["target"] == "gate2"]
        ref = {k: np.sort([v for x in planets if (v := x.get(k)) is not None and np.isfinite(v)]) for k in rows[0]["x"]}

        def encode(x):
            return {k: int(round(100 * np.searchsorted(ref[k], v, side="right") / len(ref[k])))
                    for k, v in x.items() if v is not None and np.isfinite(v) and len(ref[k])}
    else:
        encode = state
    out_rows = [{"toi": r["toi"], "split": r["label"]["split"], "target": r["label"]["target"], "state": encode(r["x"])}
                for r in rows]
    FOLDER.mkdir(parents=True, exist_ok=True)
    name = "rows.json" if encoding == "raw" else f"rows-{encoding}.json"
    (FOLDER / name).write_text(json.dumps(out_rows), encoding="utf-8")
    print(f"{len(out_rows)} rows, {len(rows[0]['x'])} checks each, in {FOLDER / name}; e.g. {json.dumps(out_rows[0]['state'])[:200]}")


def compare(*encodings: str) -> None:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    rows = [r for r in json.loads((FOLDER / "rows.json").read_text(encoding="utf-8"))
            if r["split"] == "val" and r["target"] in ("gate2", "reject")]
    runs = {f"Glimmer {e}": json.loads((FOLDER / ("preds.json" if e == "raw" else f"preds-{e}.json")).read_text(encoding="utf-8"))
            for e in ("raw", *encodings)}
    runs |= {
            "XGBoost + sector": json.loads(Path("data/finetune/system1/preds-x-sector.json").read_text(encoding="utf-8"))}
    y = np.array([r["target"] == "gate2" for r in rows])
    for name, p in runs.items():
        s = np.array([p[r["toi"]] for r in rows])
        line = f"{name:<17} val AUC {roc_auc_score(y, s):.4f}"
        for lost in (0.01, 0.02, 0.05):
            cut = np.quantile(s[y], lost)
            line += f" | {lost:.0%} lost: FPs caught {int((s[~y] < cut).sum())}/{(~y).sum()}"
        print(line)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    {"rows": make_rows, "compare": compare}[sys.argv[1]](*sys.argv[2:])
