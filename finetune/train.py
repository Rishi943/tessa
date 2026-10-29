# QLoRA fine-tune of Muse Glimmer 30B as Tessa's vetter, with Unsloth, on Modal. Settings follow Unsloth's
# Glimmer guide and its LoRA hyperparameter guide (see FINETUNE.md). Kaggle's 2xT4 tops out near 1,024
# tokens; a dossier's input alone is ~1,600, so this runs on one H100.
#   modal run finetune/train.py::download                    # once: the 4-bit base into the weights volume (CPU)
#   modal run finetune/train.py::train --data data/finetune/sft   # train.jsonl + val.jsonl; adapter into tessa-ft
#   modal run --detach finetune/train.py::train --data data/finetune/sft_v6 --detach   # the same, laptop-proof
#   modal volume get tessa-ft glimmer-vetter data/finetune/adapter
import json
from pathlib import Path

import modal

BASE = "unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit"
MAX_SEQ = 6144  # high-strength dossiers run ~4,300 tokens with the prompt (measured 2026-10-28)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")  # nvcc for JIT kernels
    .uv_pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
    .uv_pip_install("unsloth", "torchao>=0.16.0", "transformers==5.15.0", "trl==0.22.2", "datasets", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
weights = modal.Volume.from_name("tessa-hf-cache", create_if_missing=True)
out = modal.Volume.from_name("tessa-ft", create_if_missing=True)
app = modal.App("tessa-finetune")


@app.function(image=image, volumes={"/root/.cache/huggingface": weights}, timeout=60 * 60)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download(BASE)
    weights.commit()


@app.function(image=image, gpu="H100", volumes={"/root/.cache/huggingface": weights, "/out": out},
              timeout=6 * 60 * 60)
def run(train_rows: list[dict], val_rows: list[dict], epochs: float = 2.0, rank: int = 16, lr: float = 2e-4,
        dry: bool = False):
    from unsloth import FastModel  # first, so it patches transformers
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastModel.from_pretrained(model_name=BASE, max_seq_length=MAX_SEQ, load_in_4bit=True)
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=rank, lora_alpha=rank, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=3407,
    )

    def text(rows):
        return Dataset.from_list([{"text": tokenizer.apply_chat_template(r["messages"], tokenize=False)
                                   .removeprefix(tokenizer.bos_token)} for r in rows])
    train_ds, val_ds = text(train_rows), text(val_rows)
    # the tokenizer is Glimmer's multimodal processor: a positional argument would be read as an image
    lengths = sorted(len(tokenizer(text=[t])["input_ids"][0]) for t in train_ds["text"])
    print(f"train {len(train_ds)}, val {len(val_ds)}; tokens median {lengths[len(lengths) // 2]}, "
          f"max {lengths[-1]}, over {MAX_SEQ}: {sum(n > MAX_SEQ for n in lengths)}", flush=True)

    steps_per_epoch = max(1, len(train_ds) // 16)
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds, eval_dataset=val_ds,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        args=SFTConfig(
            dataset_text_field="text", max_length=MAX_SEQ, packing=False,  # trl 0.22.2 has no loss_type
            per_device_train_batch_size=2, gradient_accumulation_steps=8,  # effective batch 16
            num_train_epochs=epochs, max_steps=5 if dry else -1,
            learning_rate=lr, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
            optim="adamw_8bit", seed=3407, bf16=True, logging_steps=5,
            eval_strategy="steps", eval_steps=max(5, steps_per_epoch // 4), save_strategy="steps",
            save_steps=max(5, steps_per_epoch // 4), save_total_limit=3, load_best_model_at_end=True,
            metric_for_best_model="eval_loss", greater_is_better=False,
            output_dir="/out/checkpoints", report_to="none",
        ),
    )
    # Loss on Tessa's whole turn: the reasoning (to=self) and the dossier (to=user), not the prompt.
    trainer = train_on_responses_only(trainer, instruction_part="<|start|>user<|message|>",
                                      response_part="<|start|>assistant")
    # Check the mask on one example: the trained part must start at Tessa's reasoning.
    labels = trainer.train_dataset[0]["labels"]
    first = next(i for i, x in enumerate(labels) if x != -100)
    print("trained part starts:", repr(tokenizer.decode(trainer.train_dataset[0]["input_ids"][first:first + 12])), flush=True)

    stats = trainer.train()
    print(json.dumps(stats.metrics), flush=True)
    history = [h for h in trainer.state.log_history]
    name = "glimmer-vetter-dry" if dry else "glimmer-vetter"
    model.save_pretrained(f"/out/{name}")
    tokenizer.save_pretrained(f"/out/{name}")
    Path(f"/out/{name}/history.json").write_text(json.dumps(history, indent=1))
    out.commit()
    print(f"peak memory {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB", flush=True)
    return history


@app.local_entrypoint()
def train(data: str = "data/finetune/sft", epochs: float = 2.0, rank: int = 16, lr: float = 2e-4, dry: bool = False,
          detach: bool = False):
    rows = {s: [json.loads(line) for line in Path(data, f"{s}.jsonl").read_text(encoding="utf-8").splitlines()]
            for s in ("train", "val")}
    if detach:  # with `modal run --detach`: survives the laptop disconnecting; the history lands in the volume
        call = run.spawn(rows["train"], rows["val"], epochs, rank, lr, dry)
        print(f"spawned {call.object_id}; history in tessa-ft glimmer-vetter/history.json when it ends")
        return
    history = run.remote(rows["train"], rows["val"], epochs, rank, lr, dry)
    Path(data, "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    evals = [h for h in history if "eval_loss" in h]
    print("eval loss:", [(h["step"], round(h["eval_loss"], 4)) for h in evals])
