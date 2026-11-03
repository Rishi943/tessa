"""Voice video/script.json with Breeze (a running breeze-server on 127.0.0.1:8080) in the user's cloned voice.

  python video/voice.py [--seed 42] [--voice me-telescope] [--only SCENE_ID]

One request per sentence; takes go to video/vo/SEED/SCENE-N.wav. Then joins them with set pauses into
video/vo/SEED/full.wav and writes timings.json (each scene's and sentence's start and length, seconds), which the
animation is fitted to. Script -> audio -> animation.
"""
import argparse
import json
import struct
import urllib.parse
import urllib.request
import uuid
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
RATE = 24000
GAP_LINE, GAP_SCENE = 0.3, 0.6
DIRECTION = ("An energetic, warm young Indian man, a science YouTuber telling a story he loves: curious and clear, "
             "with natural rises and falls in pitch, and unhurried pauses at commas.")


def speak(text: str, voice: str, seed: int, direction: str = DIRECTION) -> bytes:
    fields = {"text": text, "instruction": direction, "voice_id": voice, "seed": str(seed), "cfg_scale": "1.0"}
    b = uuid.uuid4().hex
    body = b"".join(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode() for k, v in fields.items())
    body += f"--{b}--\r\n".encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/audio/speech", data=body,
                                 headers={"content-type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def write(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--voice", default="me-telescope")
    ap.add_argument("--only")
    ap.add_argument("--direction", default=DIRECTION)
    ap.add_argument("--tag", help="output folder name (default: the seed)")
    a = ap.parse_args()
    script = json.loads((HERE / "script.json").read_text(encoding="utf-8"))
    out = HERE / "vo" / (a.tag or str(a.seed))
    out.mkdir(parents=True, exist_ok=True)
    full, timings, t = b"", [], 0.0
    for si, scene in enumerate(script["scenes"]):
        if si:
            full += b"\0\0" * int(GAP_SCENE * RATE); t += GAP_SCENE
        rec = {"id": scene["id"], "start": round(t, 2), "lines": []}
        for li, text in enumerate(scene["lines"]):
            f = out / f"{scene['id']}-{li}.wav"
            if not f.exists() or a.only == scene["id"]:
                write(f, speak(text, a.voice, a.seed, a.direction))
                print(f"{f.name}", flush=True)
            with wave.open(str(f)) as w:
                pcm = w.readframes(w.getnframes())
            if li:
                full += b"\0\0" * int(GAP_LINE * RATE); t += GAP_LINE
            n = len(pcm) / 2 / RATE
            rec["lines"].append({"text": text, "start": round(t, 2), "dur": round(n, 2), "wpm": round(len(text.split()) / n * 60)})
            full += pcm; t += n
        rec["dur"] = round(t - rec["start"], 2)
        timings.append(rec)
    write(out / "full.wav", full)
    (out / "timings.json").write_text(json.dumps({"total": round(t, 2), "scenes": timings}, indent=1), encoding="utf-8")
    for s in timings:
        print(f"{s['id']:16s} {s['start']:6.1f} s  +{s['dur']:5.1f} s")
    print(f"total {t:.1f} s ({int(t // 60)}:{t % 60:04.1f})")


if __name__ == "__main__":
    main()
