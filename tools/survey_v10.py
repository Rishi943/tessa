"""Survey re-vet with vetting v10 (#12): Glimmer makes the planet call on every survey gate-1 pass that code does not settle.

Stages (each resumes; run in order):
  select   which of the 785 signals reach the model (agent-bound, or rejected only by the System 1 cut) and can still pass
           (no HARD_ROWS row, rank >= FLOOR_RANK); writes data/survey_v10/runset.json and the vet dirs data/vet/{survey,batch2}_v10
  tools    the four cached-tool answers v10 reads for each run-set star, measured locally (CPU pool); centroid_all where the
           neighbours/centroid row would flag
  glimmer  the fine-tuned vetter on Modal with those answers; dossiers go to data/dossiers/TIC.glimmer-vetter-v10.v10.json
  gate2    pipeline.gate2 on both vet dirs (novelty, test, seasons)
Run: python tools/survey_v10.py STAGE [--jobs N] [--limit N]
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import gate2, system1, tools  # noqa: E402
from pipeline.paths import DATA  # noqa: E402

WORK = DATA / "survey_v10"
BATCHES = {"survey": ("survey_results.csv", "lc"), "batch2": ("batch2_results.csv", "lc_batch2")}
MODEL = "glimmer-vetter-v10"
FOUR = ("sector_depths_by_source", "rotation", "transit_depths", "stellar_radius")


def vet_files(batch: str):
    for p in sorted((DATA / "vet" / batch).glob("[0-9]*.json")):
        yield p, json.loads(p.read_text(encoding="utf-8"))


def s1_only(c: dict) -> bool:
    return c["code_verdict"] == "reject" and bool(c["reasons"]) and all("System 1" in r for r in c["reasons"])


def select() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    run, tally = [], {}
    for batch in BATCHES:
        out = DATA / "vet" / f"{batch}_v10"
        out.mkdir(exist_ok=True)
        for p, c in vet_files(batch):
            only = s1_only(c)
            reaches = c["code_verdict"] == "agent" or only
            ev = gate2.evidence(c["tic"])
            can_pass = reaches and not gate2.outside_planet_range(c, ev, "v10")
            if can_pass:
                c["code_verdict"] = "agent"
                run.append({"tic": c["tic"], "batch": batch, "s1_only": only})
            key = (batch, "model" if can_pass else "code" if not reaches else "floor/hard-row")
            tally[key] = tally.get(key, 0) + 1
            (out / p.name).write_text(json.dumps(c, indent=1), encoding="utf-8")
    (WORK / "runset.json").write_text(json.dumps(run), encoding="utf-8")
    print(tally, "->", len(run), "signals to the model")


def runset(limit: int | None = None) -> list[dict]:
    return json.loads((WORK / "runset.json").read_text(encoding="utf-8"))[:limit]


def one_tool(tic: int, names: tuple) -> str:
    for n in names:
        tools.run(n, tic)
    return f"{tic} {','.join(names)}"


def do_tools(jobs: int, limit: int | None) -> None:
    rs = runset(limit)
    todo, need_cen = [], []
    for r in rs:
        ev, c = gate2.evidence(r["tic"]), json.loads((DATA / "vet" / r["batch"] / f"{r['tic']}.json").read_text(encoding="utf-8"))
        cen = c.get("centroid") or {}
        want = (cen.get("verdict") == "off_target" or (c["neighbours"].get("plausible_within_1px") and cen.get("verdict") != "on_target"))
        if want and "centroid_all" not in ev:
            need_cen.append(r["tic"])
        miss = tuple(n for n in FOUR[:3] if n not in ev)
        if miss:
            todo.append((r["tic"], miss))
    print(f"{len(todo)} stars need local tools, {len(need_cen)} need centroid_all", flush=True)
    with ProcessPoolExecutor(jobs, mp_context=get_context("spawn")) as pool:
        futs = [pool.submit(one_tool, t, m) for t, m in todo]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                f.result()
            except Exception as e:
                print("tool error:", str(e)[:150], flush=True)
            if n % 25 == 0:
                print(f"{n}/{len(futs)}", flush=True)
    # stellar_radius: one Gaia query for every star, then the (cheap) arithmetic here
    need = [r["tic"] for r in rs if "stellar_radius" not in gate2.evidence(r["tic"])]
    ids = {}
    for t in need:
        c = tools.locate(t)[0]
        if c["star"].get("source_id"):
            ids[t] = int(c["star"]["source_id"])
    for i in range(0, len(ids), 200):
        tools.gaia_params(list(ids.values())[i:i + 200])
    for t in need:
        tools.run("stellar_radius", t)
    # centroid_all last: TESScut is the slow, rate-limited one
    with ProcessPoolExecutor(3, mp_context=get_context("spawn")) as pool:
        futs = [pool.submit(one_tool, t, ("centroid_all",)) for t in need_cen]
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print("centroid error:", str(e)[:150], flush=True)


def views(rs: list[dict], score: bool = True) -> dict[int, str]:
    from finetune.pilot import EXTRA, FOLDER
    jobs = []
    for r in rs:
        c = json.loads((DATA / "vet" / r["batch"] / f"{r['tic']}.json").read_text(encoding="utf-8"))
        clue = c.get("clues") or {}
        x = system1.features(c)
        x.update(system1.clue_features({"checks": c}, clue, clue.get("gaia") or {}, clue.get("eb") or {}, system1.CLUE_GROUPS))
        extra = {k: v for k, v in x.items() if EXTRA.match(k)}
        jobs.append({"toi": str(r["tic"]), "checks": c, "extra": extra, "score": round(c["system1"]["planet_rank"], 2)})
    path = WORK / "view_jobs.json"
    path.write_text(re.sub(r"\bNaN\b|-?\bInfinity\b", "null", json.dumps(jobs, default=float)), encoding="utf-8")
    res = subprocess.run(["node", "finetune/pilot_view.js", "v10", str(path)], capture_output=True, text=True, encoding="utf-8", cwd=ROOT)
    if res.returncode:
        raise RuntimeError(res.stderr[-800:])
    return {int(v["toi"]): v["user"] for v in json.loads(res.stdout)}


def answers(tic: int) -> dict:
    ev = gate2.evidence(tic)
    a = {k: ev[k] for k in ("sector_depths_by_source", "rotation", "transit_depths", "stellar_radius") if k in ev}
    cen = ev.get("centroid_all")
    a["centroid_all"] = cen if cen and not cen.get("error") else {"not_run": True}
    return a


def lenient(text: str, parse) -> dict | None:
    """pilot.parse, then again with the opening brace Glimmer sometimes drops between two reasons put back."""
    return parse(text) or parse(re.sub(r'\}\s*,\s*"check"', '},{"check"', text))


def do_glimmer(jobs: int, limit: int | None) -> None:
    from finetune.pilot import SCORE_NOTE, invented, investigate, parse, prompt_text, tool_specs
    from finetune.teacher import load_env
    load_env()
    rs = runset(limit)
    users = views(rs)
    prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
    specs = tool_specs(prompt)
    save = WORK / "glimmer"
    save.mkdir(exist_ok=True)

    def work(r):
        tic = r["tic"]
        f = save / f"{tic}.json"
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
        a = answers(tic)
        try:
            rec = investigate(prompt, users[tic], specs, a, model="glimmer-vetter")
        except Exception as e:
            print(f"{tic}: {str(e)[:200]}", flush=True)
            return None
        rec |= {"tic": tic, "user": users[tic], "answers": {k: a[k] for k in set(rec["tools"]) if k in a}}
        d = lenient(rec["text"], parse)
        rec["invented"] = invented(rec, prompt)
        f.write_text(json.dumps(rec), encoding="utf-8")
        if d:
            d["reasons"] = [{"check": "", "why": "", **r} for r in d["reasons"] if isinstance(r, dict)]  # gate2 reads both
            (DATA / "dossiers" / f"{tic}.{MODEL}.v10.json").write_text(json.dumps(
                {"model": MODEL, "prompt": "v10", "raw": rec["text"], "dossier": {"open_questions": [], "public_summary": "", **d},
                 "invented": rec["invented"], "tools": rec["tools"]}), encoding="utf-8")
        return rec

    done = 0
    with ThreadPoolExecutor(jobs) as pool:
        for rec in pool.map(work, rs):
            done += 1
            if done % 20 == 0:
                print(f"{done}/{len(rs)}", flush=True)
    print("glimmer done", flush=True)


def do_gate2() -> None:
    procs = []
    for batch, (results, lc) in BATCHES.items():
        procs.append(subprocess.Popen([sys.executable, "-m", "pipeline.gate2", str(DATA / "vet" / f"{batch}_v10"), "--results",
                                       str(DATA / results), "--lc", str(DATA / "cache" / lc), "--model", MODEL, "--prompt", "v10"],
                                      cwd=ROOT, stdout=open(WORK / f"gate2_{batch}.log", "w", encoding="utf-8"), stderr=subprocess.STDOUT))
    for p in procs:
        p.wait()
    for batch in BATCHES:
        print(batch, (WORK / f"gate2_{batch}.log").read_text(encoding="utf-8").strip().splitlines()[-1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["select", "tools", "glimmer", "gate2"])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    {"select": select, "tools": lambda: do_tools(a.jobs, a.limit), "glimmer": lambda: do_glimmer(a.jobs, a.limit),
     "gate2": do_gate2}[a.stage]()


if __name__ == "__main__":
    main()
