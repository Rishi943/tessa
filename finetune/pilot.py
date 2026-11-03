"""Pilot for prompt v10: Spark Open makes the planet call and picks its tools. With --score, System 1's rank is in its
input as one piece of evidence (SCORE_NOTE), as in every v10 run after the pilot.
Train signals only; the test split is never read.
  python -m finetune.pilot run ROUND [--n 20] [--seed S] [--version v10] [--same R] [--score]   fresh signals (8 planets, 8 FPs, 4 APCs), run, report
  python -m finetune.pilot report ROUND                                    the report again from the saved traces
  python -m finetune.pilot show ROUND TOI                                  one trace
  python -m finetune.pilot oracle                                          the v10 gate 2 rule against the train signals
  python -m finetune.pilot traces --batch N [--jobs 8]                     the next N train signals of the full plan, v10 with
                                                                           --score, full messages kept in data/finetune/pilot/traces"""
import argparse
import json
import random
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pipeline.paths import DATA, ROOT

from finetune.teacher import MAX_CALLS, FOLDER as TEACHER, load_env, post, verdict

FOLDER = DATA / "finetune" / "pilot"
VET = DATA / "finetune" / "vet_s1"
TRACES = FOLDER / "traces"
PROMPTS = ROOT / "agent" / "prompts"
PRICE = {"input": 1.25, "output": 4.25}  # $ per million tokens: agent/models.js sparkOpen
MIX = {"gate2": 8, "reject": 8, "flag": 4}
EXTRA = re.compile(r"^(fit_|star_density_penalty|sector_oe_|sector_sec_half_|secondary_over_bump|secondary_phase_offset|"
                   r"variability_chi2_dof|excess_noise_sig|gaia_eb_|ipd_|radius_tic_over_flame|rv_)")
TOOL_NAMES = ("centroid_all", "sector_depths_by_source", "transit_depths", "stellar_radius", "rotation")


def prompt_text(version: str) -> str:
    return (PROMPTS / f"vetting.{version}.md").read_text(encoding="utf-8")


def tool_specs(prompt: str) -> list[dict]:
    """The tools' descriptions as the prompt words them: its '- name: text' lines under Tools."""
    found = {m[1]: m[2] for m in re.finditer(r"^- (\w+): (.+)$", prompt, re.M) if m[1] in TOOL_NAMES}
    return [{"type": "function", "function": {"name": k, "description": found[k],
                                              "parameters": {"type": "object", "properties": {}}}} for k in TOOL_NAMES]


def used_before() -> set[str]:
    path = FOLDER / "used.json"
    return {t for tois in json.loads(path.read_text(encoding="utf-8")).values() for t in tois} if path.exists() else set()


def system_one_only(reasons: list[str]) -> bool:
    return bool(reasons) and all("System 1" in r for r in reasons)


def candidates() -> dict[str, list[dict]]:
    """Train signals with tool results, by target: eligible are the agent's and the code's System-1-only rejects."""
    tools = {p.stem.removeprefix("tools_"): p for p in TEACHER.rglob("tools_*.json")}
    out = {"gate2": [], "reject": [], "flag": []}
    for f in sorted(VET.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        c = d["checks"]
        if d["label"]["split"] != "train" or d["toi"] not in tools or d["label"]["target"] not in out:
            continue
        only = c["code_verdict"] == "reject" and system_one_only(c["reasons"])
        if c["code_verdict"] != "agent" and not only:
            continue
        out[d["label"]["target"]].append({"toi": d["toi"], "d": d, "s1_only": only})
    return out


def pick(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    cand, used = candidates(), used_before()
    for k in cand:
        cand[k] = [x for x in cand[k] if x["toi"] not in used]
    want = {k: round(v * n / 20) for k, v in MIX.items()}
    chosen = rng.sample(cand["gate2"], min(want["gate2"], len(cand["gate2"])))
    chosen += rng.sample(cand["flag"], min(want["flag"], len(cand["flag"])))
    only = [x for x in cand["reject"] if x["s1_only"]]
    rest = [x for x in cand["reject"] if not x["s1_only"]]
    k = min(want["reject"] // 2, len(only))
    fps = rng.sample(only, k)
    chosen += fps + rng.sample(rest, min(want["reject"] - k, len(rest)))
    return chosen


SCORE_NOTE = ("The input also holds xgboost.planet_rank: a separate scorer, trained on TESS candidates whose fate follow-up "
              "settled, read these checks. 0.5 means as planet-like as the median known planet; below 0.1 only one planet in ten scores this low, and false "
              "positives often do, so gate2 there needs specific evidence the scorer could have missed, such as a tool's answer. "
              "Treat it as one strong piece of evidence, not the verdict: you decide, and you may disagree with it when the "
              "evidence says so.")


def same_as(round_: str) -> list[dict]:
    """The signals of an earlier round, in that round's files."""
    tois = {f.stem for f in (FOLDER / f"r{round_}").glob("*.json")}
    if not tois:
        raise SystemExit(f"no signals in round {round_}")
    cand = [x for v in candidates().values() for x in v if x["toi"] in tois]
    if len(cand) != len(tois):
        raise SystemExit(f"round {round_}: {len(tois) - len(cand)} signals are not candidates any more")
    return sorted(cand, key=lambda x: x["toi"])


def views(chosen: list[dict], version: str, score: bool = False) -> dict[str, str]:
    rows = {r["toi"]: r["state"] for r in json.loads((DATA / "finetune" / "s1_glimmer" / "rows.json").read_text(encoding="utf-8"))}
    jobs = [{"toi": x["toi"], "checks": x["d"]["checks"],
             "extra": {k: v for k, v in rows.get(x["toi"], {}).items() if EXTRA.match(k)},
             **({"score": round(x["d"]["checks"]["system1"]["planet_rank"], 2)} if score else {})} for x in chosen]
    path = FOLDER / "view_jobs.json"
    path.write_text(re.sub(r"\bNaN\b|-?\bInfinity\b", "null", json.dumps(jobs)), encoding="utf-8")  # null: not measured
    r = subprocess.run(["node", "finetune/pilot_view.js", version, str(path)], capture_output=True, text=True,
                       encoding="utf-8", cwd=ROOT)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])
    users = {v["toi"]: v["user"] for v in json.loads(r.stdout)}
    for x in chosen:  # nothing that names the star, the outcome or the TOI
        u = users[x["toi"]]
        bad = [w for w in (re.findall(r"\d{12,}", u) + re.findall(r"TIC|TOI|comment|TFOP|source_id|\"ra\"|\"dec\"|upcoming|\"t0\"", u)
                           + ([x["toi"]] if x["toi"] in u else []) + re.findall(r"system1|code_verdict", u)) if w]
        if bad:
            raise RuntimeError(f"{x['toi']}: leak in the input: {bad[:3]}")
    return users


def investigate(system: str, user: str, specs: list[dict], answers: dict, model: str = "spark-open",
                keep: bool = False, on_call=None) -> dict:
    """The tool loop. With keep, the record also holds the complete messages list: every assistant turn with its
    reasoning and tool_calls, every tool result, the final answer."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    full = list(messages)
    trace = list(messages)  # the whole conversation for training, with the reasoning the API doesn't take back
    body = {"max_tokens": 16384, "tools": specs}
    used, tokens, thinking = [], {"input": 0, "output": 0}, []
    t0 = time.time()
    while True:
        if len(used) >= MAX_CALLS:
            body["tool_choice"] = "none"
        c = post(model, {**body, "messages": messages})
        tokens["input"] += c["usage"]["prompt_tokens"]
        tokens["output"] += c["usage"]["completion_tokens"]
        msg = c["choices"][0]["message"]
        thinking.append(msg.get("reasoning_content") or msg.get("reasoning") or "")
        calls = msg.get("tool_calls") or []
        trace.append({"role": "assistant", "content": msg.get("content") or "", "reasoning": thinking[-1],
                      **({"tool_calls": calls} if calls else {})})
        if not calls or len(used) >= MAX_CALLS:
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            used.append(name)
            if on_call:  # tools/demo.py --events shows each call as it happens
                on_call(name, call["function"].get("arguments"), answers.get(name))
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": json.dumps(answers.get(name, {"error": f"no tool called {name}"}))})
            trace.append(messages[-1])
    return {"ms": int((time.time() - t0) * 1000), "tokens": tokens, "tools": used, "text": msg.get("content") or "", "messages": trace,
            "thinking": "\n---\n".join(t for t in thinking if t), "finish_reason": c["choices"][0]["finish_reason"]}


def tool_answers(toi: str) -> dict:
    t = next(TEACHER.rglob(f"tools_{toi}.json"))
    answers = {k: v for k, v in json.loads(t.read_text(encoding="utf-8")).items() if k in TOOL_NAMES}
    cen = next(TEACHER.rglob(f"centroid_{toi}.json"), None)
    answers["centroid_all"] = json.loads(cen.read_text(encoding="utf-8"))["centroid_all"] if cen else {"not_run": True}
    if answers["centroid_all"].get("error"):
        answers["centroid_all"] = {"not_run": True}
    return answers


def numbers_in(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\w./+])-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", text, re.I)]


def invented(rec: dict, prompt: str) -> list[float]:
    """agent/dossier.js inventedNumbers, with the tool answers counted as inputs too."""
    d = parse(rec["text"])
    if d is None:
        return []
    known = numbers_in(rec["user"] + prompt + json.dumps(rec["answers"]))
    near = lambda a, b: abs(a - b) <= max(0.15, 0.03 * abs(b))  # noqa: E731
    cited = [x for r in d.get("reasons", []) if isinstance(r, dict) for x in numbers_in(f"{r.get('value')} {r.get('why')}")]
    return [x for x in cited if not (x == int(x) and abs(x) <= 10) and not any(near(x, k) or near(x, 100 * k) for k in known)]


def parse(text: str) -> dict | None:
    """The dossier JSON, tolerating a tool-call block before it: the last balanced {...} with a valid verdict."""
    for m in reversed(list(re.finditer(r"\{(?:[^{}]|\{[^{}]*\})*\}", text, re.S))):
        try:
            d = json.loads(m.group(0))
        except ValueError:
            continue
        if d.get("verdict") in ("reject", "flag", "gate2") and isinstance(d.get("reasons"), list):
            return d
    return None


def report(round_: str) -> str:
    folder = FOLDER / f"r{round_}"
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(folder.glob("*.json"))]
    prompt = prompt_text(recs[0]["version"])
    for r in recs:
        r["verdict"] = verdict(r["text"])
        r["invented"] = invented(r, prompt)
    n = len(recs)
    calls = Counter(t for r in recs for t in r["tools"])
    conf = Counter((r["target"], r["verdict"]) for r in recs)
    tin, tout = sum(r["tokens"]["input"] for r in recs), sum(r["tokens"]["output"] for r in recs)
    cost = (tin * PRICE["input"] + tout * PRICE["output"]) / 1e6
    lines = [f"round {round_}: {n} signals, {sum(r['verdict'] is None for r in recs)} parse failures",
             f"tool calls per signal: {sum(len(r['tools']) for r in recs) / n:.2f}; "
             f"{sum(bool(r['tools']) for r in recs)}/{n} used at least one tool",
             "tool mix: " + (", ".join(f"{k} {v}" for k, v in calls.most_common()) or "none"),
             "target -> verdict:"]
    for t in ("gate2", "reject", "flag"):
        lines.append(f"  {t:<7}" + "  ".join(f"{v}={conf[(t, v)]}" for v in ("gate2", "flag", "reject", None)))
    lines += [f"invented numbers: {sum(len(r['invented']) for r in recs)} in {sum(bool(r['invented']) for r in recs)} signals",
              f"tokens: {tin} in, {tout} out; about ${cost:.3f}", "wrong verdicts:"]
    for r in recs:
        if r["verdict"] != r["target"]:
            d = parse(r["text"]) or {}
            why = " | ".join(f"{x.get('check')}: {x.get('why')}" for x in d.get("reasons", []) if isinstance(x, dict))
            lines.append(f"  {r['toi']} target {r['target']} verdict {r['verdict']} tools {r['tools']} :: "
                         f"{(why or r['text'])[:200]}")
    text = "\n".join(lines)
    (folder / "summary.txt").write_text(text + "\n", encoding="utf-8")
    return text


def run(round_: str, n: int, seed: int | None, version: str, same: str | None = None, score: bool = False) -> None:
    load_env()
    folder = FOLDER / f"r{round_}"
    folder.mkdir(parents=True, exist_ok=True)
    chosen = same_as(same) if same else pick(n, int(round_) if seed is None else seed)
    users = views(chosen, version, score)
    prompt = prompt_text(version) + (("\n\n" + SCORE_NOTE) if score else "")
    specs = tool_specs(prompt)
    if not same:
        path = FOLDER / "used.json"
        used = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        used[round_] = [x["toi"] for x in chosen]
        path.write_text(json.dumps(used), encoding="utf-8")
    print(f"round {round_}: {len(chosen)} signals " + str(dict(Counter(x['d']['label']['target'] for x in chosen))), flush=True)

    def work(x):
        answers = tool_answers(x["toi"])
        try:
            rec = investigate(prompt, users[x["toi"]], specs, answers)
        except Exception as e:
            return print(f"{x['toi']}: {e}", flush=True)
        rec |= {"toi": x["toi"], "target": x["d"]["label"]["target"], "version": version, "score": score, "user": users[x["toi"]],
                "answers": {k: answers[k] for k in set(rec["tools"]) if k in answers}}
        (folder / f"{x['toi']}.json").write_text(json.dumps(rec), encoding="utf-8")

    with ThreadPoolExecutor(4) as pool:
        list(pool.map(work, chosen))
    if any(folder.glob("*.json")):
        print(report(round_))


def trace_plan() -> list[str]:
    """Every FP and APC with 500 planets, shuffled so each batch has the same mix; fixed on the first run."""
    path = TRACES / "plan.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    TRACES.mkdir(parents=True, exist_ok=True)
    cand, rng = candidates(), random.Random(0)
    tois = [x["toi"] for x in rng.sample(cand["gate2"], min(500, len(cand["gate2"])))]
    tois += [x["toi"] for k in ("reject", "flag") for x in cand[k]]
    rng.shuffle(tois)
    path.write_text(json.dumps(tois), encoding="utf-8")
    print(f"plan: {len(tois)} signals: 500 planets at most, {len(cand['reject'])} FPs, {len(cand['flag'])} APCs", flush=True)
    return tois


def trace_report(planned: int) -> None:
    prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in TRACES.glob("*.json") if f.name != "plan.json"]
    kept, right, total = Counter(), Counter(), Counter()
    for r in recs:
        v = verdict(r["text"])
        total[r["target"]] += 1
        right[r["target"]] += v == r["target"]
        kept[r["target"]] += v == r["target"] and not invented(r, prompt)
    tin, tout = sum(r["tokens"]["input"] for r in recs), sum(r["tokens"]["output"] for r in recs)
    print(f"{len(recs)} of {planned} done; about ${(tin * PRICE['input'] + tout * PRICE['output']) / 1e6:.2f}; "
          f"{sum(len(r['tools']) for r in recs) / max(len(recs), 1):.2f} tool calls a signal; "
          f"{sum(bool(invented(r, prompt)) for r in recs)} with invented numbers", flush=True)
    for t in ("gate2", "reject", "flag"):
        print(f"  {t:<7} correct {right[t]}/{total[t]}, pass the training filter {kept[t]}", flush=True)


def traces(batch: int, jobs: int) -> None:
    load_env()
    tois = trace_plan()
    by_toi = {x["toi"]: x for v in candidates().values() for x in v}
    todo = [t for t in tois if not (TRACES / f"{t}.json").exists() and by_toi[t]["d"]["label"]["target"] != "flag"][:batch]  # APCs: 2 of 39 usable
    chosen = [by_toi[t] for t in todo]
    users = views(chosen, "v10", True)
    prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
    specs = tool_specs(prompt)
    stop = threading.Event()

    def work(x):
        if stop.is_set():
            return
        answers = tool_answers(x["toi"])
        try:
            rec = investigate(prompt, users[x["toi"]], specs, answers)
        except Exception as e:
            print(f"{x['toi']}: {str(e)[:200]}", flush=True)
            if "HTTP 40" in str(e) or "gave up" in str(e):  # the key or its credit: every other signal would fail too
                stop.set()
            return
        rec |= {"toi": x["toi"], "target": x["d"]["label"]["target"], "version": "v10", "score": True, "user": users[x["toi"]],
                "answers": {k: answers[k] for k in set(rec["tools"]) if k in answers}}
        (TRACES / f"{x['toi']}.json").write_text(json.dumps(rec), encoding="utf-8")

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(work, chosen))
    if stop.is_set():
        print("stopped: the API refused the key or the credit", flush=True)
    trace_report(len(tois))


FORCED = FOLDER / "traces_forced"
FORCE_NOTE = "\n\nFor this signal, before you decide, call rotation and stellar_radius, as well as anything else that is open."


def forced(controls: int, jobs: int) -> None:
    """Planets and FPs the traces got wrong so far, plus some they got right, again with rotation and stellar_radius
    required: do those two change the verdict? Into traces_forced/; the note goes in the record's force field."""
    load_env()
    FORCED.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in TRACES.glob("*.json") if f.name != "plan.json"]
    recs = [r for r in recs if r["target"] in ("gate2", "reject") and not (FORCED / f"{r['toi']}.json").exists()]
    wrong = [r for r in recs if verdict(r["text"]) != r["target"]]
    right = random.Random(0).sample([r for r in recs if verdict(r["text"]) == r["target"]
                                     and not {"rotation", "stellar_radius"} <= set(r["tools"])], controls)
    prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
    specs = tool_specs(prompt)
    by_toi = {x["toi"]: x for v in candidates().values() for x in v}
    print(f"{len(wrong)} wrong, {len(right)} controls", flush=True)

    def work(r):
        answers = tool_answers(r["toi"])
        try:
            rec = investigate(prompt + FORCE_NOTE, r["user"], specs, answers)
        except Exception as e:
            return print(f"{r['toi']}: {str(e)[:200]}", flush=True)
        rec |= {"toi": r["toi"], "target": by_toi[r["toi"]]["d"]["label"]["target"], "version": "v10", "score": True, "user": r["user"],
                "force": FORCE_NOTE, "before": verdict(r["text"]), "answers": {k: answers[k] for k in set(rec["tools"]) if k in answers}}
        (FORCED / f"{r['toi']}.json").write_text(json.dumps(rec), encoding="utf-8")

    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(work, wrong + right))
    done = [json.loads(f.read_text(encoding="utf-8")) for f in FORCED.glob("*.json")]
    for t in ("gate2", "reject"):
        for label, group in (("was wrong", [r for r in done if r["target"] == t and r["before"] != t]),
                             ("was right", [r for r in done if r["target"] == t and r["before"] == t])):
            print(f"  {t} {label}: now correct {sum(verdict(r['text']) == t for r in group)}/{len(group)}", flush=True)


def show(round_: str, toi: str) -> None:
    r = json.loads((FOLDER / f"r{round_}" / f"{toi}.json").read_text(encoding="utf-8"))
    print(f"{toi} target {r['target']} verdict {verdict(r['text'])} tools {r['tools']} tokens {r['tokens']} {r['ms'] / 1000:.0f} s")
    print("--- input\n" + r["user"][:1500] + ("..." if len(r["user"]) > 1500 else ""))
    print("--- thinking\n" + r["thinking"][:3000])
    print("--- answer\n" + r["text"])


def oracle() -> None:
    """v10 rule on every train signal: a HARD_ROWS row always blocks; v9 outputs are untouched by the new argument."""
    from pipeline import gate2

    def v9(c, ev):  # outside_planet_range as it was before the version argument
        rows = gate2.table_rows(c, ev)
        s1 = c.get("system1")
        if s1 is None:
            return rows
        rank = s1.get("planet_rank") or 0
        if rank >= gate2.PASS_RANK:
            return [r for r in rows if r.startswith(gate2.HARD_ROWS)]
        return [f"system1: planet rank {rank:.2f}, below the {gate2.PASS_RANK:g} that passes without follow-up "
                f"({1 - rank:.0%} of known planets score higher)"] + rows

    total = touched = bad = same = 0
    for f in sorted(VET.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d["label"]["split"] != "train":
            continue
        c, ev = d["checks"], d.get("evidence")
        total += 1
        hard = [r for r in gate2.table_rows(c, ev) if r.startswith(gate2.HARD_ROWS)]
        out = gate2.outside_planet_range(c, ev, "v10")
        touched += bool(hard)
        bad += (bool(hard) and not out) or out != hard
        if total <= 50:
            same += gate2.outside_planet_range(c, ev) == v9(c, ev) == gate2.outside_planet_range(c, ev, "v9")
    print(f"{total} train signals; {touched} have a HARD_ROWS row (none can pass under v10); violations of the v10 rule: {bad}")
    print(f"v9 outputs unchanged on the first {min(total, 50)} records: {same}/{min(total, 50)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("round")
    r.add_argument("--n", type=int, default=20)
    r.add_argument("--seed", type=int)
    r.add_argument("--version", default="v10")
    r.add_argument("--same", help="rerun the signals of this earlier round; used.json is left alone")
    r.add_argument("--score", action="store_true", help="give the model xgboost.planet_rank and a prompt paragraph on it")
    sub.add_parser("report").add_argument("round")
    s = sub.add_parser("show")
    s.add_argument("round")
    s.add_argument("toi")
    sub.add_parser("oracle")
    t = sub.add_parser("traces")
    t.add_argument("--batch", type=int, default=200)
    t.add_argument("--jobs", type=int, default=8)
    f = sub.add_parser("forced")
    f.add_argument("--controls", type=int, default=20)
    f.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "forced":
        forced(a.controls, a.jobs)
    elif a.cmd == "run":
        run(a.round, a.n, a.seed, a.version, a.same, a.score)
    elif a.cmd == "report":
        print(report(a.round))
    elif a.cmd == "show":
        show(a.round, a.toi)
    elif a.cmd == "traces":
        traces(a.batch, a.jobs)
    else:
        oracle()


if __name__ == "__main__":
    main()
