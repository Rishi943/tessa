# Glimmer dossiers generated inside Modal, so a flaky laptop connection can't break the run: a CPU
# container sends the prompts to the deployed endpoint (serve/glimmer.py) and keeps every reply in the
# tessa-ft volume. Detached, it keeps going after the laptop disconnects.
#   node finetune/traces.js --splits test --export data/finetune/prompts_test.jsonl
#   modal run --detach finetune/remote_traces.py --prompts data/finetune/prompts_test.jsonl --name base-v5-test
#   modal volume get tessa-ft traces_raw/base-v5-test data/finetune/raw   # later, on wifi
#   node finetune/traces.js --import data/finetune/raw/base-v5-test
import json
from pathlib import Path

import modal

app = modal.App("tessa-traces")
out = modal.Volume.from_name("tessa-ft", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("httpx==0.28.1")


@app.function(image=image, volumes={"/out": out}, secrets=[modal.Secret.from_name("glimmer-key")],
              timeout=6 * 60 * 60)
def run(packed: bytes, name: str, url: str, jobs: int = 48):
    import asyncio
    import gzip
    import os
    import time

    import httpx

    prompts = [json.loads(line) for line in gzip.decompress(packed).decode("utf-8").splitlines() if line.strip()]
    folder = Path("/out/traces_raw") / name
    folder.mkdir(parents=True, exist_ok=True)
    todo = [p for p in prompts if not (folder / f"{p['id']}.json").exists()]
    print(f"{len(todo)} of {len(prompts)} to write", flush=True)
    headers = {"Authorization": f"Bearer {os.environ['GLIMMER_API_KEY']}"}
    done, failed, started = 0, 0, time.time()

    async def one(client, p):
        nonlocal done, failed
        body = {"model": "muse-glimmer", "max_tokens": 16384,
                "messages": [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}]}
        for attempt in range(4):
            try:
                t0 = time.time()
                r = await client.post(f"{url}/v1/chat/completions", json=body, headers=headers, timeout=1800)
                r.raise_for_status()
                c = r.json()
                msg = c["choices"][0]["message"]
                rec = {k: p[k] for k in p if k not in ("system",)}
                rec.update(ms=int((time.time() - t0) * 1000), finish_reason=c["choices"][0]["finish_reason"],
                           tokens={"input": c["usage"]["prompt_tokens"], "output": c["usage"]["completion_tokens"]},
                           thinking=msg.get("reasoning_content") or msg.get("reasoning") or "", text=msg.get("content") or "")
                (folder / f"{p['id']}.json").write_text(json.dumps(rec))
                done += 1
                return
            except Exception as e:  # cold starts and 5xx: wait and retry
                print(f"{p['id']}: {e.__class__.__name__} {str(e)[:120]}", flush=True)
                await asyncio.sleep(30 * (attempt + 1))
        failed += 1

    async def main():
        sem = asyncio.Semaphore(jobs)
        async with httpx.AsyncClient(follow_redirects=True) as client:  # Modal redirects long requests (303)
            async def guarded(p):
                async with sem:
                    await one(client, p)
                    if (done + failed) % 25 == 0:
                        print(f"{done + failed}/{len(todo)} ({failed} failed), {(time.time() - started) / 60:.1f} min", flush=True)
                        out.commit()
            await asyncio.gather(*(guarded(p) for p in todo))

    asyncio.run(main())
    out.commit()
    print(f"done: {done} written, {failed} failed, {(time.time() - started) / 60:.1f} min", flush=True)


@app.local_entrypoint()
def main(prompts: str, name: str, jobs: int = 48):
    import os
    url = os.environ.get("GLIMMER_URL") or next(
        line.split("=", 1)[1].strip() for line in Path(".env").read_text(encoding="utf-8").splitlines()
        if line.startswith("GLIMMER_URL="))
    import gzip
    packed = gzip.compress(Path(prompts).read_bytes())  # a small upload: mobile data drops big ones
    print(f"{len(packed) / 1e6:.2f} MB of prompts")
    # spawn, not remote: the call then belongs to the detached app, and a dropped laptop connection
    # (which stopped the first run at 303/338) can no longer end it
    call = run.spawn(packed, name, url.rstrip("/"), jobs)
    print(f"spawned {call.object_id}; the app keeps running after this exits")
