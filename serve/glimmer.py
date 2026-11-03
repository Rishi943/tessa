# Muse Glimmer 30B (open weights) behind an OpenAI-compatible vLLM endpoint on Modal, for the
# agent's dossiers and investigations. One L40S (48 GB) runs the FP8 build (32.8 GB). For big batches,
# GLIMMER_GPU=H100 at deploy time: twice the price per hour, but its 80 GB hold twice the requests at once
# and it writes each token faster, so a batch costs no more and ends sooner. The container scales to zero
# five minutes after the last request, so a batch pays only for the minutes it runs.
#   modal run serve/glimmer.py::download   # once: weights into a Modal volume (CPU, no GPU time)
#   modal run serve/glimmer.py::adapter --repo HF_USER/glimmer-vetter-v10   # once: Glimmer v10's published adapter
#   modal deploy serve/glimmer.py          # prints the URL; set GLIMMER_URL to it in .env
#   GLIMMER_GPU=H100 modal deploy serve/glimmer.py   # same URL, the bigger GPU
# The endpoint checks GLIMMER_API_KEY from the Modal secret "glimmer-key" (same value in .env).
import os
import subprocess

import modal

MODEL = "RedHatAI/Muse-Glimmer-30B-FP8-block"  # block-scaled FP8 weights; vision tower stays BF16
VLLM = "0.32.0"  # the muse_glimmer tool-call and reasoning parsers need 0.28+
GPU = os.environ.get("GLIMMER_GPU", "L40S")
SEQS = 128 if GPU.startswith("H100") else 64  # requests batched at once: what the KV cache left after the weights holds
# H100s can all be taken; then the next 80 GB card, which holds as many requests (Modal tries the list in order)
GPUS = ["H100", "A100-80GB"] if GPU.startswith("H100") else GPU

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM}")
    # FlashInfer's top-k/top-p sampler compiles itself on first use and needs nvcc, which the slim
    # image lacks; the engine then dies during warm-up. vLLM's own PyTorch sampler needs nothing.
    # GLIMMER_GPU goes in too: the container re-reads this file, and its SEQS must match the deploy's GPU.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0", "GLIMMER_GPU": GPU})
)
weights = modal.Volume.from_name("tessa-hf-cache", create_if_missing=True)
compiled = modal.Volume.from_name("tessa-vllm-cache", create_if_missing=True)
ft = modal.Volume.from_name("tessa-ft", create_if_missing=True)  # finetune/train.py's adapter
VOLUMES = {"/root/.cache/huggingface": weights, "/root/.cache/vllm": compiled}

app = modal.App("tessa-glimmer")


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    weights.commit()


def vllm(*extra):
    subprocess.Popen([
        "vllm", "serve", MODEL, "--served-model-name", "muse-glimmer",
        "--host", "0.0.0.0", "--port", "8000", "--api-key", os.environ["GLIMMER_API_KEY"],
        # 32k covers a dossier or a five-call investigation; the full 131k would only shrink the KV cache.
        "--max-model-len", "32768", "--max-num-seqs", str(SEQS),
        "--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer",
        "--reasoning-parser", "muse_glimmer", "--generation-config", "auto", *extra,
    ])


@app.function(image=image, gpu=GPUS, volumes=VOLUMES, secrets=[modal.Secret.from_name("glimmer-key")],
              scaledown_window=5 * 60, timeout=60 * 60, max_containers=2)
@modal.concurrent(max_inputs=SEQS)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    vllm()


# The fine-tuned vetter: the same FP8 weights plus the LoRA adapter, asked for as model "glimmer-vetter"
# ("muse-glimmer" still answers as the base). Its own endpoint, so the base keeps serving while it starts.
@app.function(image=image, gpu=GPUS, volumes={**VOLUMES, "/ft": ft}, secrets=[modal.Secret.from_name("glimmer-key")],
              scaledown_window=5 * 60, timeout=60 * 60, max_containers=2)
@modal.concurrent(max_inputs=SEQS)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def vetter():
    # v10 is the published adapter (modal run serve/glimmer.py::adapter fetches it); older ones load only if present
    mods = ["glimmer-vetter-v10=/ft/glimmer-vetter-v10"] + [f"{n}=/ft/{n}" for n in ("glimmer-vetter",) if os.path.isdir(f"/ft/{n}")]
    vllm("--enable-lora", "--lora-modules", *mods, "--max-lora-rank", "16")


# Glimmer v10's published weights from Hugging Face into the adapter volume, so anyone can serve the same model:
#   modal run serve/glimmer.py::adapter --repo HF_USER/glimmer-vetter-v10
@app.function(image=image, volumes={"/ft": ft}, timeout=30 * 60)
def adapter(repo: str):
    from huggingface_hub import snapshot_download

    snapshot_download(repo, local_dir="/ft/glimmer-vetter-v10")
    ft.commit()
