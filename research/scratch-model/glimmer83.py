"""Glimmer v10 on the survey dips the net would release from the 5% floor (XGBoost set them aside).

Same as tools/survey_v10.py's tools + glimmer stages, except: the rank Glimmer reads (xgboost.planet_rank) is
the net's, and every output stays in work/glimmer83/ (no dossiers, runset or view_jobs in data/ are touched).
The cached tool answers (rotation, depths, stellar radius, centroid_all) go to the normal evidence cache.

Run from the repo root: python research/scratch-model/glimmer83.py select|tools|glimmer|report [--limit N] [--jobs N]
"""
import argparse
import copy
import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from pipeline import gate2, system1, tools  # noqa: E402
from pipeline.paths import DATA  # noqa: E402

OUT = HERE / "work" / "glimmer83"


def vet(r):
    return json.loads((DATA / "vet" / f"{r['batch']}_v10" / f"{r['tic']}.json").read_text(encoding="utf-8"))


def with_net_rank(c, r):
    c = copy.deepcopy(c)
    c.setdefault("system1", {}).update(planet_rank=float(r["nn_rank"]), p_planet=float(r["nn_p"]))
    return c


def select():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(HERE / "work" / "survey_scores.csv", encoding="utf-8")))
    picks, hard = [], 0
    for r in rows:
        if r["code"] == "agent" and float(r["xgb_rank"]) < 0.05 <= float(r["nn_rank"]):
            c = with_net_rank(vet(r), r)
            why = gate2.outside_planet_range(c, gate2.evidence(int(r["tic"])), "v10")
            if why:
                hard += 1
                continue
            picks.append({"tic": int(r["tic"]), "batch": r["batch"], "nn_rank": float(r["nn_rank"]), "nn_p": float(r["nn_p"]),
                          "xgb_rank": float(r["xgb_rank"])})
    (OUT / "runset.json").write_text(json.dumps(picks, indent=1), encoding="utf-8")
    print(f"{len(picks)} dips go to Glimmer with the net's rank ({hard} more stopped by a hard row)")


def runset(limit=None):
    return json.loads((OUT / "runset.json").read_text(encoding="utf-8"))[:limit]


def do_tools(jobs, limit):
    from survey_v10 import FOUR, one_tool
    rs = runset(limit)
    todo, need_cen = [], []
    for r in rs:
        ev, c = gate2.evidence(r["tic"]), vet(r)
        cen = c.get("centroid") or {}
        if (cen.get("verdict") == "off_target" or (c["neighbours"].get("plausible_within_1px") and cen.get("verdict") != "on_target")) \
                and "centroid_all" not in ev:
            need_cen.append(r["tic"])
        miss = tuple(n for n in FOUR[:3] if n not in ev)
        if miss:
            todo.append((r["tic"], miss))
    print(f"{len(todo)} stars need local tools, {len(need_cen)} need centroid_all", flush=True)
    with ProcessPoolExecutor(jobs, mp_context=get_context("spawn")) as pool:
        for n, f in enumerate(as_completed([pool.submit(one_tool, t, m) for t, m in todo]), 1):
            try:
                f.result()
            except Exception as e:
                print("tool error:", str(e)[:150], flush=True)
            if n % 10 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    need = [r["tic"] for r in rs if "stellar_radius" not in gate2.evidence(r["tic"])]
    ids = {t: int(tools.locate(t)[0]["star"]["source_id"]) for t in need if tools.locate(t)[0]["star"].get("source_id")}
    for i in range(0, len(ids), 200):
        tools.gaia_params(list(ids.values())[i:i + 200])
    for t in need:
        tools.run("stellar_radius", t)
    with ProcessPoolExecutor(3, mp_context=get_context("spawn")) as pool:
        for f in as_completed([pool.submit(one_tool, t, ("centroid_all",)) for t in need_cen]):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print("centroid error:", str(e)[:150], flush=True)
    print("tools done", flush=True)


def views(rs):
    from finetune.pilot import EXTRA
    jobs = []
    for r in rs:
        c = vet(r)
        clue = c.get("clues") or {}
        x = system1.features(c)
        x.update(system1.clue_features({"checks": c}, clue, clue.get("gaia") or {}, clue.get("eb") or {}, system1.CLUE_GROUPS))
        jobs.append({"toi": str(r["tic"]), "checks": c, "extra": {k: v for k, v in x.items() if EXTRA.match(k)},
                     "score": round(r["nn_rank"], 2)})  # the net's rank where production puts XGBoost's
    path = OUT / "view_jobs.json"
    path.write_text(re.sub(r"\bNaN\b|-?\bInfinity\b", "null", json.dumps(jobs, default=float)), encoding="utf-8")
    res = subprocess.run(["node", "finetune/pilot_view.js", "v10", str(path)], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if res.returncode:
        raise RuntimeError(res.stderr[-800:])
    return {int(v["toi"]): v["user"] for v in json.loads(res.stdout)}


def do_glimmer(jobs, limit):
    from finetune.pilot import SCORE_NOTE, invented, investigate, prompt_text, tool_specs
    from finetune.teacher import load_env
    from survey_v10 import answers
    load_env()
    rs = runset(limit)
    users = views(rs)
    prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
    specs = tool_specs(prompt)
    (OUT / "records").mkdir(parents=True, exist_ok=True)

    def work(r):
        f = OUT / "records" / f"{r['tic']}.json"
        if f.exists():
            return "cached"
        a = answers(r["tic"])
        try:
            rec = investigate(prompt, users[r["tic"]], specs, a, model="glimmer-vetter")
        except Exception as e:
            return f"{r['tic']}: {str(e)[:200]}"
        rec |= {"tic": r["tic"], "user": users[r["tic"]], "answers": {k: a[k] for k in set(rec["tools"]) if k in a}}
        rec["invented"] = invented(rec, prompt)
        f.write_text(json.dumps(rec), encoding="utf-8")
        return "ok"

    with ThreadPoolExecutor(jobs) as pool:
        for n, msg in enumerate(pool.map(work, rs), 1):
            if msg not in ("ok", "cached"):
                print(msg, flush=True)
            if n % 10 == 0:
                print(f"{n}/{len(rs)}", flush=True)
    print("glimmer done", flush=True)


def report():
    from finetune.pilot import parse
    from survey_v10 import lenient
    out, tally = [], {}
    for r in runset():
        f = OUT / "records" / f"{r['tic']}.json"
        if not f.exists():
            tally["not run"] = tally.get("not run", 0) + 1
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        d = lenient(rec["text"], parse)
        v = d.get("verdict") if d else "unreadable"
        tally[v] = tally.get(v, 0) + 1
        out.append({**r, "verdict": v, "reasons": [x.get("why", "") for x in (d or {}).get("reasons", []) if isinstance(x, dict)][:3]})
    (OUT / "verdicts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("Glimmer verdicts:", tally)
    for o in out:
        if o["verdict"] == "gate2":
            print(f"  PASS TIC {o['tic']} ({o['batch']}): net rank {o['nn_rank']:.3f}, XGBoost rank {o['xgb_rank']:.3f}")


def do_gate2(only=None, name="gate2"):
    """pipeline.gate2.decide on each Glimmer pass (novelty, ephemeris, seasons), as for the survey; nothing written to data/.
    only: a set of TICs to (re)run; results go to work/glimmer83/{name}.json."""
    from finetune.pilot import parse
    from pipeline.paths import batches
    from pipeline.vet import signal_from_results
    from survey_v10 import lenient
    b = batches()
    out = []
    for r in runset():
        f = OUT / "records" / f"{r['tic']}.json"
        if not f.exists() or (only and r["tic"] not in only):
            continue
        rec = json.loads(f.read_text(encoding="utf-8"))
        d = lenient(rec["text"], parse)
        if not d or d.get("verdict") != "gate2":
            continue
        d["reasons"] = [{"check": "", "why": "", **x} for x in d["reasons"] if isinstance(x, dict)]
        dossier = {"model": "glimmer-vetter-v10", "prompt": "v10", "raw": rec["text"],
                   "dossier": {"open_questions": [], "public_summary": "", **d}}
        res, lc = b[r["batch"]]
        c = with_net_rank(vet(r), r)
        try:
            status, reasons, extra = gate2.decide(c, dossier, signal_from_results(r["tic"], res), lc)
        except Exception as e:
            status, reasons, extra = "error", [str(e)[:200]], {}
        out.append({**r, "status": status, "reasons": reasons})
        print(f"TIC {r['tic']} ({r['batch']}): {status}" + (f"  {'; '.join(reasons)[:220]}" if reasons else ""), flush=True)
    (OUT / f"{name}.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    tally = {}
    for o in out:
        tally[o["status"]] = tally.get(o["status"], 0) + 1
    print("gate 2:", tally)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["select", "tools", "glimmer", "report", "gate2"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--tics", help="comma-separated TICs: gate2 reruns only these, into gate2_retry.json")
    a = ap.parse_args()
    only = {int(t) for t in a.tics.split(",")} if a.tics else None
    {"select": select, "tools": lambda: do_tools(a.jobs, a.limit), "glimmer": lambda: do_glimmer(a.jobs, a.limit),
     "report": report, "gate2": lambda: do_gate2(only, "gate2_retry" if only else "gate2")}[a.stage]()
