# Muse Glimmer 30B (open weights) behind an OpenAI-compatible vLLM endpoint on Modal, for the
# agent's dossiers and investigations. One L40S (48 GB) runs the FP8 build (32.8 GB): H100s need a
# card on the Modal account. The container scales to zero five minutes after the last request, so a
# batch pays only for the minutes it runs.
#   modal run serve/glimmer.py::download   # once: weights into a Modal volume (CPU, no GPU time)
#   modal deploy serve/glimmer.py          # prints the URL; set GLIMMER_URL to it in .env
# The endpoint checks GLIMMER_API_KEY from the Modal secret "glimmer-key" (same value in .env).
import subprocess

import modal

MODEL = "RedHatAI/Muse-Glimmer-30B-FP8-block"  # block-scaled FP8 weights; vision tower stays BF16
VLLM = "0.32.0"  # the muse_glimmer tool-call and reasoning parsers need 0.28+

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"vllm=={VLLM}")
    # FlashInfer's top-k/top-p sampler compiles itself on first use and needs nvcc, which the slim
    # image lacks; the engine then dies during warm-up. vLLM's own PyTorch sampler needs nothing.
    .env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})
)
weights = modal.Volume.from_name("tessa-hf-cache", create_if_missing=True)
compiled = modal.Volume.from_name("tessa-vllm-cache", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": weights, "/root/.cache/vllm": compiled}

app = modal.App("tessa-glimmer")


@app.function(image=image, volumes=VOLUMES, timeout=60 * 60)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL)
    weights.commit()


@app.function(image=image, gpu="L40S", volumes=VOLUMES, secrets=[modal.Secret.from_name("glimmer-key")],
              scaledown_window=5 * 60, timeout=60 * 60, max_containers=2)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    import os

    subprocess.Popen([
        "vllm", "serve", MODEL, "--served-model-name", "muse-glimmer",
        "--host", "0.0.0.0", "--port", "8000", "--api-key", os.environ["GLIMMER_API_KEY"],
        # 32k covers a dossier or a five-call investigation; the full 131k would only shrink the KV cache.
        "--max-model-len", "32768", "--max-num-seqs", "64",
        "--enable-auto-tool-choice", "--tool-call-parser", "muse_glimmer",
        "--reasoning-parser", "muse_glimmer", "--generation-config", "auto",
    ])
