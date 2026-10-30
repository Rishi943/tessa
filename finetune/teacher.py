"""Teacher test: Glimmer, Spark Open and Spark 1.3 on the same dev signals, given the dossier as text, the
dossier plus a plot, or the dossier plus tools to call. It shows whether a bigger model vets better, and
whether its correct answers could teach Glimmer (Spark 1.3's can't: the Meta Model API terms forbid it).
Train and val signals only; the test split stays untouched.
  python -m finetune.teacher pick DIR [--all]      200 signals (100 planets, 100 false positives) + jobs file;
                                                   --all: jobs for every dev signal (tools only), for the oracle
  (Kaggle) python teacher_main.py [--no-plots]     plots + tool results for each, from the light curves
  python -m finetune.teacher run MODEL MODE        MODEL: glimmer | spark-open | spark-1.3; MODE: text | plots | tools
  python -m finetune.teacher score                 planets lost and FPs caught per model and mode
  python -m finetune.teacher attach DIR            the two tools prompt v8 reads, into data/finetune/vet_s1 (checks.tools)"""
import argparse
import base64
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from pipeline.paths import DATA

FOLDER = DATA / "finetune" / "teacher"
PROMPTS = DATA / "finetune" / "prompts_v7_dev.jsonl"
GLIMMER_TEXT = DATA / "finetune" / "raw" / "base-v7-dev"  # the v7 confirmation run: Glimmer's text mode
PLANET, FALSE = ("KP", "CP"), ("FP", "FA")
MAX_CALLS = 5
TOOLS = {  # the investigation tools that measure the light curve; known (it names the TOI) and predict are left out
    "sector_depths_by_source": "Transit depth per sector with the light-curve product it came from (SPOC 2-min, "
                               "TESS-SPOC, QLP). For depths that change between sectors.",
    "transit_depths": "Depth of every single transit: outliers, and how much of the signal the top three carry. "
                      "For signals that might come from a few events.",
    "stellar_radius": "Gaia DR3's radius for the star against the TIC's, with the expected transit duration and "
                      "planet size from each. For transits too long for the star.",
    "rotation": "The star's spot rotation period per sector, and whether the dip repeats at a multiple of it. For "
                "stars that swing at the signal period.",
}
PLOT_NOTE = ("\n\nThe image shows this signal's light curve in four panels: (1) folded on the period around the "
             "transit, grey points and black 30-minute bins; (2) odd transits (blue) against even ones (orange); "
             "(3) the whole orbit, where an eclipse of a companion would show near phase 0.5; (4) the depth in each "
             "TESS sector with its error. Use what you see alongside the table; quote numbers only from the table.")
TOOLS_NOTE = (f"\n\nYou may call up to {MAX_CALLS} tools before answering; each measures this signal's light "
              "curve or star and returns numbers you may then quote. Call none if the table settles it. Your final "
              "message is the JSON verdict.")
MODELS = {
    "glimmer": ("muse-glimmer", "GLIMMER_URL", "GLIMMER_API_KEY"),
    "spark-open": ("meta-models/muse-spark-open-1026", "https://inference.baseten.co", "BASETEN_API_KEY"),
    "spark-1.3": ("muse-spark-1.3", "https://api.meta.ai", "META_API_KEY"),
}


def load_env() -> None:
    for line in (DATA.parent / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"'))


def pick(job_dir: Path, every: bool) -> None:
    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if every:
        chosen = rows
    else:
        rng = random.Random(0)
        chosen = []
        for kind in (PLANET, FALSE):
            group = sorted((r for r in rows if r["disp"] in kind), key=lambda r: r["id"])
            chosen += rng.sample(group, 100)
        FOLDER.mkdir(parents=True, exist_ok=True)
        (FOLDER / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chosen), encoding="utf-8")
    jobs = []
    for r in chosen:
        d = json.loads((DATA / "finetune" / "vet" / f"{r['toi']}.json").read_text(encoding="utf-8"))
        jobs.append({"toi": r["toi"], "tic": d["tic"], "checks": {k: d["checks"][k] for k in ("star", "signal")}})
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "teacher_jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    print(f"{len(chosen)} signals ({sum(r['split'] == 'val' for r in chosen)} val), jobs in {job_dir}")


def centroid_pick(job_dir: Path, splits: list[str]) -> None:
    """The signals centroid_all is for: agent-bound, with the centroid row off target or a plausible neighbour inside
    one pixel that the centroid row doesn't clear. Writes centroid_jobs.json for centroid-job."""
    jobs = []
    for f in sorted((DATA / "finetune" / "vet_s1").glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        c = d["checks"]
        if d["label"]["split"] not in splits or c["code_verdict"] != "agent":
            continue
        verdict = (c.get("centroid") or {}).get("verdict")
        if verdict == "off_target" or ((c.get("neighbours") or {}).get("plausible_within_1px") and verdict != "on_target"):
            jobs.append({"toi": d["toi"], "tic": d["tic"], "checks": {k: c[k] for k in ("star", "signal")}})
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "centroid_jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    print(f"{len(jobs)} signals ({', '.join(splits)}), jobs in {job_dir}")


def plot(lc: dict, sig: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pipeline.vet import box_depth, phase_offset

    p, t0, dur = sig["period"], sig["t0"], sig["dur"]
    dt = phase_offset(lc["t"], p, t0)
    y = lc["y"] * 1e6  # ppm, dips negative
    near = np.abs(dt) < 3 * dur

    def binned(x, v, width):
        k = np.floor(x / width).astype(int)
        _, inv, n = np.unique(k, return_inverse=True, return_counts=True)
        return np.bincount(inv, x) / n, np.bincount(inv, v) / n

    fig, ax = plt.subplots(2, 2, figsize=(8, 5.5), dpi=100)
    h = dt * 24
    bx, by = binned(h[near], y[near], 0.5)
    pad = max(by.max() - by.min(), 1) * 0.6  # the bins set the scale: raw scatter would hide a shallow dip
    few = np.flatnonzero(near)[::max(1, near.sum() // 3000)]  # 3,000 grey points show the scatter; more only add bytes
    ax[0, 0].plot(h[few], y[few], ".", color="0.75", ms=1.5)
    ax[0, 0].plot(bx, by, "k.-", ms=4, lw=0.8)
    ax[0, 0].set(xlabel="hours from mid-transit", ylabel="flux (ppm)", title="folded",
                 ylim=(by.min() - pad, by.max() + pad))
    epoch = np.round((lc["t"] - t0) / p).astype(int)
    for parity, colour, name in ((1, "tab:blue", "odd"), (0, "tab:orange", "even")):
        m = near & (epoch % 2 == parity)
        if m.sum() > 2:
            ax[0, 1].plot(*binned(h[m], y[m], 0.5), ".-", color=colour, ms=4, lw=0.8, label=name)
    ax[0, 1].legend(fontsize=8)
    ax[0, 1].set(xlabel="hours from mid-transit", title="odd vs even")
    phase = ((lc["t"] - t0) / p + 0.25) % 1 - 0.25
    ax[1, 0].plot(*binned(phase, y, float(np.clip(dur / p / 2, 0.001, 0.005))), "k.", ms=2)
    ax[1, 0].axvline(0.5, color="tab:red", lw=0.6, ls=":")
    ax[1, 0].set(xlabel="orbital phase", ylabel="flux (ppm)", title="whole orbit")
    inside = np.abs(dt) < dur / 2
    rows = [(s, *box_depth(lc, inside & (lc["sector"] == s))) for s in np.unique(lc["sector"])]
    rows = [r for r in rows if np.isfinite(r[1])]
    ax[1, 1].errorbar([str(int(r[0])) for r in rows], [r[1] * 1e6 for r in rows], [r[2] * 1e6 for r in rows],
                      fmt="o", color="k", ms=3, lw=0.8)
    ax[1, 1].set(xlabel="TESS sector", ylabel="depth (ppm)", title="depth per sector")
    ax[1, 1].tick_params(axis="x", labelsize=6, rotation=90)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def job(out: Path, lc_root: Path, plots: bool) -> None:
    """Kaggle: each signal's plot and the result of every tool, with the light curve from the fetch job."""
    import pipeline.tools as tools
    from finetune.inputs import shards_by_tic
    from pipeline.search import detrend

    started = time.time()
    jobs = json.loads(Path("teacher_jobs.json").read_text(encoding="utf-8"))
    shards = shards_by_tic(lc_root)
    out.mkdir(parents=True, exist_ok=True)
    for n, j in enumerate(jobs, 1):
        if j["tic"] not in shards:
            print(f"{j['toi']}: no light curve", flush=True)
            continue
        with np.load(shards[j["tic"]]) as z:
            raw = {k: z[f"{j['tic']}_{k}"] for k in ("time", "flux", "sector", "source")}
        lc = detrend(raw["time"], raw["flux"], raw["sector"])
        tools.locate = lambda tic, c=j["checks"]: (c, None)
        tools.light_curve = lambda tic, c=j["checks"], r=raw, l=lc: (c, r, l)
        result = {"toi": j["toi"]}
        for name in TOOLS:
            try:
                result[name] = json.loads(json.dumps(tools.TOOLS[name](j["tic"]), default=float))
            except Exception as e:  # one tool's failure is its answer
                result[name] = {"error": f"{e.__class__.__name__}: {e}"[:200]}
        (out / f"tools_{j['toi']}.json").write_text(json.dumps(result), encoding="utf-8")
        try:
            if plots:
                plot(lc, j["checks"]["signal"], out / f"plot_{j['toi']}.png")
        except Exception as e:
            print(f"{j['toi']}: plot failed {e}", flush=True)
        if n % 100 == 0:
            print(f"{time.strftime('%H:%M:%S')} {n}/{len(jobs)} in {(time.time() - started) / 60:.0f} min", flush=True)
    print(f"done in {(time.time() - started) / 60:.0f} min", flush=True)


def centroid_job(out: Path, lc_root: Path, part: int, parts: int, max_hours: float = 11.0) -> None:
    """Kaggle: centroid_all (difference imaging on every sector, not the latest six) for the signals the neighbours
    or centroid rows flag; slow (TESScut cutouts), so in parts, each written as it finishes."""
    from finetune.inputs import shards_by_tic
    from pipeline.centroid import where

    started = time.time()
    jobs = json.loads(Path("centroid_jobs.json").read_text(encoding="utf-8"))[part::parts]
    shards = shards_by_tic(lc_root)
    out.mkdir(parents=True, exist_ok=True)
    for n, j in enumerate(jobs, 1):
        if time.time() - started > max_hours * 3600:
            print(f"stopped at the time cap after {n - 1}", flush=True)
            break
        if j["tic"] not in shards:
            continue
        with np.load(shards[j["tic"]]) as z:
            sectors = sorted({int(s) for s in z[f"{j['tic']}_sector"]})
        try:
            c = where(j["tic"], j["checks"]["signal"], max_sectors=99, star=j["checks"]["star"], sectors=sectors)
            r = {k: c.get(k) for k in ("verdict", "why", "offset_arcsec", "err_arcsec", "offset_sigma", "sectors_used",
                                       "on_star", "suspects")}
        except Exception as e:
            r = {"error": f"{e.__class__.__name__}: {e}"[:200]}
        (out / f"centroid_{j['toi']}.json").write_text(json.dumps({"toi": j["toi"], "centroid_all": r}, default=float),
                                                       encoding="utf-8")
        print(f"{time.strftime('%H:%M:%S')} {n}/{len(jobs)} {j['toi']}: {r.get('verdict', r.get('error'))}", flush=True)
    print(f"done in {(time.time() - started) / 60:.0f} min", flush=True)


def post(model: str, body: dict) -> dict:
    import httpx

    name, base, key = MODELS[model]
    base = os.environ.get(base, base).rstrip("/")
    for attempt in range(5):
        try:
            # Modal answers a request still running after 150 s with a 303 to the same URL, as often as it takes:
            # urllib calls that a loop, so httpx with no cap on redirects
            with httpx.Client(follow_redirects=True, max_redirects=1000, timeout=1800) as client:
                r = client.post(f"{base}/v1/chat/completions", json={"model": name, **body},
                                headers={"Authorization": f"Bearer {os.environ[key]}"})
            if r.status_code in (400, 401, 403, 404):
                raise PermissionError(f"HTTP {r.status_code}: {r.text[:300]}")
            r.raise_for_status()
            return r.json()
        except PermissionError as e:
            raise RuntimeError(str(e))
        except Exception as e:
            print(f"  retry {attempt + 1}: {e.__class__.__name__} {str(e)[:100]}", flush=True)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("gave up after 5 tries")


def one(model: str, mode: str, p: dict, tool_results: dict, png: Path) -> dict:
    system = p["system"] if model == "glimmer" else p["system"].replace("\n\nReasoning strength: low.", "")
    user = p["user"]
    if mode == "plots":
        system += PLOT_NOTE
        user = [{"type": "text", "text": p["user"]},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode()}}]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    body = {"max_tokens": 16384}
    if mode == "tools":
        messages[0]["content"] += TOOLS_NOTE
        body["tools"] = [{"type": "function", "function": {"name": k, "description": v,
                                                           "parameters": {"type": "object", "properties": {}}}}
                         for k, v in TOOLS.items()]
    t0, used, tokens = time.time(), [], {"input": 0, "output": 0}
    while True:
        if mode == "tools" and len(used) >= MAX_CALLS:
            body["tool_choice"] = "none"
        c = post(model, {**body, "messages": messages})
        tokens["input"] += c["usage"]["prompt_tokens"]
        tokens["output"] += c["usage"]["completion_tokens"]
        msg = c["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls or mode != "tools":
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            used.append(name)
            answer = tool_results.get(name, {"error": f"no tool called {name}"})
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(answer)})
    return {"id": p["id"], "toi": p["toi"], "split": p["split"], "disp": p["disp"], "model": model, "mode": mode,
            "ms": int((time.time() - t0) * 1000), "tokens": tokens, "tools": used,
            "finish_reason": c["choices"][0]["finish_reason"], "text": msg.get("content") or "",
            "thinking": msg.get("reasoning_content") or msg.get("reasoning") or ""}


def run(model: str, mode: str, jobs: int) -> None:
    load_env()
    prompts = [json.loads(line) for line in (FOLDER / "prompts.jsonl").read_text(encoding="utf-8").splitlines()]
    results = FOLDER / "kaggle"
    out = FOLDER / f"{model}-{mode}"
    out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in prompts if not (out / f"{p['id']}.json").exists()]
    print(f"{model} {mode}: {len(todo)} of {len(prompts)} to run", flush=True)
    done = 0

    def work(p):
        nonlocal done
        tool_file = next(results.rglob(f"tools_{p['toi']}.json"), None)
        png = next(results.rglob(f"plot_{p['toi']}.png"), None)
        if mode != "text" and (tool_file is None or png is None):
            return print(f"{p['id']}: no Kaggle results", flush=True)
        try:
            rec = one(model, mode, p, json.loads(tool_file.read_text(encoding="utf-8")) if tool_file else {}, png)
        except Exception as e:
            return print(f"{p['id']}: {e}", flush=True)
        (out / f"{p['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        done += 1
        if done % 20 == 0:
            print(f"{time.strftime('%H:%M:%S')} {done}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(work, todo))
    print(f"done: {done} of {len(todo)}", flush=True)


def verdict(text: str) -> str | None:
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        v = json.loads(body[body.index("{"):body.rindex("}") + 1]).get("verdict")
    except (ValueError, AttributeError):
        return None
    return v if v in ("reject", "flag", "gate2") else None


def score() -> None:
    prompts = {json.loads(line)["id"]: json.loads(line)
               for line in (FOLDER / "prompts.jsonl").read_text(encoding="utf-8").splitlines()}
    runs = {"glimmer-text": [json.loads((GLIMMER_TEXT / f"{i}.json").read_text(encoding="utf-8")) | {"disp": p["disp"]}
                             for i, p in prompts.items() if (GLIMMER_TEXT / f"{i}.json").exists()]}
    for folder in sorted(FOLDER.glob("*-*")):
        if folder.is_dir() and folder.name != "glimmer-text":
            runs[folder.name] = [json.loads(f.read_text(encoding="utf-8")) for f in folder.glob("*.json")]
    print(f"{'run':<18} {'n':>4} {'planets lost':>13} {'planets on':>11} {'FPs caught':>11} {'unparsed':>9} "
          f"{'tool calls':>10} {'out tokens':>10}")
    for name, recs in runs.items():
        v = [(r["disp"], verdict(r.get("text") or "")) for r in recs]
        planets = [x for d, x in v if d in PLANET]
        fps = [x for d, x in v if d in FALSE]
        calls = np.mean([len(r.get("tools") or []) for r in recs]) if recs else 0
        out_tok = np.mean([(r.get("tokens") or {}).get("output", 0) for r in recs]) if recs else 0
        print(f"{name:<18} {len(recs):>4} {planets.count('reject'):>6}/{len(planets):<6} "
              f"{planets.count('gate2'):>4}/{len(planets):<6} {fps.count('reject'):>4}/{len(fps):<6} "
              f"{sum(x is None for _, x in v):>9} {calls:>10.1f} {out_tok:>10.0f}")


EVIDENCE_KEYS = ("sector_depths_by_source", "rotation", "centroid_all")  # what vet's for_agent saves for gate 2


def attach(folder: Path) -> None:
    """The Kaggle tools_ and centroid_ results into vet_s1: the full results as the record's evidence (gate 2's,
    what production saves to data/evidence/TIC.json), the short form as checks.tools (the prompt's)."""
    from pipeline.tools import compact

    vet = DATA / "finetune" / "vet_s1"
    n = 0
    for p in [*folder.rglob("tools_*.json"), *folder.rglob("centroid_*.json")]:
        t = json.loads(p.read_text(encoding="utf-8"))
        path = vet / f"{t['toi']}.json"
        if not path.exists() or "error" in t.get("centroid_all", {}):
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        d["evidence"] = d.get("evidence", {}) | {k: v for k, v in t.items() if k in EVIDENCE_KEYS}
        d["checks"]["tools"] = compact(d["evidence"])
        path.write_text(json.dumps(d), encoding="utf-8")
        n += 1
    print(f"{n} results attached in {vet}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("pick")
    k.add_argument("dir", type=Path)
    k.add_argument("--all", action="store_true")
    j = sub.add_parser("job")
    j.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    j.add_argument("--lc", type=Path, default=Path(os.environ.get("KAGGLE_INPUTS", "/kaggle/input")))
    j.add_argument("--no-plots", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("model", choices=list(MODELS))
    r.add_argument("mode", choices=("text", "plots", "tools"))
    r.add_argument("--jobs", type=int, default=16)
    sub.add_parser("score")
    cj = sub.add_parser("centroid-job")
    cj.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    cj.add_argument("--lc", type=Path, default=Path(os.environ.get("KAGGLE_INPUTS", "/kaggle/input")))
    cj.add_argument("--part", type=int, default=0)
    cj.add_argument("--parts", type=int, default=1)
    cp = sub.add_parser("centroid-pick")
    cp.add_argument("dir", type=Path)
    cp.add_argument("--split", nargs="+", default=["train", "val"])
    sub.add_parser("attach").add_argument("dir", type=Path)
    a = ap.parse_args()
    if a.cmd == "pick":
        pick(a.dir, a.all)
    elif a.cmd == "job":
        job(a.out, a.lc, not a.no_plots)
    elif a.cmd == "centroid-job":
        centroid_job(a.out, a.lc, a.part, a.parts)
    elif a.cmd == "centroid-pick":
        centroid_pick(a.dir, a.split)
    elif a.cmd == "attach":
        attach(a.dir)
    elif a.cmd == "run":
        run(a.model, a.mode, a.jobs)
    else:
        score()


if __name__ == "__main__":
    main()
