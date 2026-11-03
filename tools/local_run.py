"""The site's "Run a star" page, on this machine: tools/demo.py's steps streamed to http://localhost:5173/run.

  cd site && npm run local          starts this, the Worker (Ask Tessa) and the site together

Serves 127.0.0.1:8788 only (the site's dev server proxies /local to it):
  GET /local/health      {"glimmer": live or replay, "meta": Ask Tessa has a key, "stars": the bundled stars}
  GET /local/run?tic=N   server-sent events: tools/demo.py --events, one run at a time

Keys come from the repo's .env (GLIMMER_VETTER_URL and GLIMMER_API_KEY for a live Glimmer; site/.dev.vars holds
META_API_KEY for Ask Tessa). Without Glimmer's, the run replays its recorded answer and the page says so.
"""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
PORT = 8788
busy = threading.Lock()


def env() -> dict:
    e = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                e.setdefault(k.strip(), v.strip().strip('"'))
    return e


def has(name: str, text: str) -> bool:
    return any(line.split("=", 1)[0].strip() == name and line.split("=", 1)[1].strip() for line in text.splitlines() if "=" in line)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def send_json(self, body: dict, status: int = 200) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.send_header("access-control-allow-origin", "*")  # the page on localhost:5173 calls this port directly
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/local/health":
            e = env()
            dev = ROOT / "site" / ".dev.vars"
            stars = json.loads((ROOT / "demo" / "data" / "demo.json").read_text(encoding="utf-8"))["stars"]
            return self.send_json({"glimmer": "live" if e.get("GLIMMER_VETTER_URL") and e.get("GLIMMER_API_KEY") else "replay",
                                   "meta": dev.exists() and has("META_API_KEY", dev.read_text(encoding="utf-8")),
                                   "stars": [{"tic": int(t), "note": m["note"]} for t, m in stars.items()]})
        if url.path == "/local/run":
            tic = parse_qs(url.query).get("tic", [""])[0]
            if not tic.isdigit():
                return self.send_json({"error": "tic must be a number"}, 400)
            if not busy.acquire(blocking=False):
                return self.send_json({"error": "a run is already going"}, 409)
            try:
                self.stream(int(tic))
            finally:
                busy.release()
            return
        self.send_json({"error": "not found"}, 404)

    def stream(self, tic: int) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-store")
        self.send_header("access-control-allow-origin", "*")
        self.end_headers()
        e = env()
        cmd = [sys.executable, "-u", str(ROOT / "tools" / "demo.py"), str(tic), "--events"]
        if e.get("GLIMMER_VETTER_URL") and e.get("GLIMMER_API_KEY"):
            cmd.append("--live")
        p = subprocess.Popen(cmd, cwd=ROOT, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", bufsize=1)
        # a comment line every 15 s keeps the stream open while a step works quietly (TRICERATOPS can take minutes)
        alive = threading.Event()

        def beat():
            while not alive.wait(15):
                try:
                    self.wfile.write(b": still working\n\n"); self.wfile.flush()
                except OSError:
                    return
        threading.Thread(target=beat, daemon=True).start()
        try:
            for raw in p.stdout:
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                msg = line[2:] if line.startswith("@@") else json.dumps({"type": "text", "text": line})
                self.wfile.write(f"data: {msg}\n\n".encode())
                self.wfile.flush()
            code = p.wait()
            self.wfile.write(f"data: {json.dumps({'type': 'exit', 'code': code})}\n\n".encode())
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):  # the page was closed: stop the run
            p.kill()
        finally:
            alive.set()


if __name__ == "__main__":
    print(f"Tessa local runs on http://127.0.0.1:{PORT} (open the site at http://localhost:5173/run)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
