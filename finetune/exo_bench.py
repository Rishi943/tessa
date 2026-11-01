"""Tessa end to end (Glimmer v10 + the code's hard rows and rank floor) against NASA ExoMiner++ 2.0 on the TOIs both scored.

A signal is a Tessa pass when code did not reject it, Glimmer said gate2 and gate2.outside_planet_range(v10) is empty (the same
call valgate counts). ExoMiner++ passes at its published 0.5 threshold. Planets are follow-up gate2 outcomes, false positives
are rejects; APCs ('flag') are left out like everywhere else.

Every shared TOI, train and val too, shows how the fine-tune holds up: the report keeps the splits apart, because train
signals are what Glimmer learned from (its training rows are reported on their own) and val tuned the floor. Only test is
clean. AUC needs a score and Glimmer gives a verdict, so the score is the tier (pass > flag > reject) with System 1's
out-of-fold P(planet) breaking ties inside a tier; the verdict alone is shown beside it.

Run: python -m finetune.exo_bench run [--jobs 8]   Glimmer on the shared train signals (Modal; resumes; val and test are saved)
     python -m finetune.exo_bench                  the report
"""
import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.metrics import roc_auc_score

from finetune.compare import exominer
from finetune.pilot import SCORE_NOTE, investigate, prompt_text, tool_answers, tool_specs, views
from finetune.system1 import OUT as S1
from finetune.teacher import load_env, verdict
from finetune.valgate import MODEL, val_signals
from pipeline import gate2
from pipeline.paths import DATA

BOOT = 5000
SEED = 20261101
SAVED = {s: DATA / "finetune" / d for s, d in (("train", "traingate"), ("val", "valgate"), ("test", "testgate"))}


def run(jobs: int) -> None:
    load_env()
    exo = exominer()
    sigs = [s for s in val_signals("train") if s["reaches"] and s["toi"] in exo]
    SAVED["train"].mkdir(exist_ok=True)
    print(f"{len(sigs)} shared train signals reach the model", flush=True)
    users = views(sigs, "v10", True)
    prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
    specs = tool_specs(prompt)

    def work(s):
        f = SAVED["train"] / f"{s['toi']}.json"
        if f.exists():
            return
        answers = tool_answers(s["toi"])
        try:
            rec = investigate(prompt, users[s["toi"]], specs, answers, model=MODEL)
        except Exception as e:
            print(f"{s['toi']}: {str(e)[:200]}", flush=True)
            return
        rec |= {"toi": s["toi"], "target": s["d"]["label"]["target"]}
        f.write_text(json.dumps(rec), encoding="utf-8")

    with ThreadPoolExecutor(jobs) as pool:
        for n, _ in enumerate(pool.map(work, sigs), 1):
            if n % 25 == 0:
                print(f"{n}/{len(sigs)}", flush=True)


def tier(s: dict, split: str) -> int | None:
    """2 a Tessa pass, 1 stopped short of it (flag, or gate2 stopped by a hard row or the floor), 0 rejected; None: no run."""
    if not s["reaches"]:
        return 0
    f = SAVED[split] / f"{s['toi']}.json"
    if not f.exists():
        return None
    v = verdict(json.loads(f.read_text(encoding="utf-8"))["text"])
    if v == "reject" or v is None:
        return 0
    return 2 if v == "gate2" and not gate2.outside_planet_range(s["d"]["checks"], s["d"].get("evidence"), "v10") else 1


def ci(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    """95% bootstrap interval of mean(a) - mean(b), paired."""
    idx = rng.integers(0, len(a), (BOOT, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def auc_ci(y: np.ndarray, a: np.ndarray, b: np.ndarray, rng) -> tuple[float, float]:
    out = []
    for _ in range(BOOT // 5):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def table(name: str, rows: list[dict], rng) -> None:
    p, f = [r for r in rows if r["planet"]], [r for r in rows if not r["planet"]]
    if not p or not f:
        return
    y = np.array([r["planet"] for r in rows])

    def col(k):
        return np.array([r[k] for r in rows], float)

    print(f"\n{name}: {len(rows)} TOIs, {len(p)} planets, {len(f)} FPs")
    for label, key in (("Tessa e2e (Glimmer v10)", "tessa"), ("XGBoost rank>=0.32 (v9)", "xgb"), ("ExoMiner++ at 0.5", "exo")):
        a, b = sum(r[key] for r in p), sum(r[key] for r in f)
        print(f"  {label:26} planets {a:>3}/{len(p)} ({a / len(p):4.0%})  FPs through {b:>3}/{len(f)} ({b / len(f):5.1%})")
    tessa, em = roc_auc_score(y, col("score")), roc_auc_score(y, col("exo_score"))
    print(f"  AUC: Tessa {tessa:.3f}, verdict alone {roc_auc_score(y, col('tier')):.3f}, "
          f"System 1 {roc_auc_score(y, col('s1')):.3f}, ExoMiner++ {em:.3f}")
    lo, hi = auc_ci(y, col("score"), col("exo_score"), rng)
    rp = ci(np.array([r["tessa"] for r in p], float), np.array([r["exo"] for r in p], float), rng)
    rf = ci(np.array([r["tessa"] for r in f], float), np.array([r["exo"] for r in f], float), rng)
    print(f"  Tessa minus ExoMiner++: AUC {tessa - em:+.3f} [{lo:+.3f}, {hi:+.3f}], "
          f"planets [{rp[0]:+.1%}, {rp[1]:+.1%}], FPs [{rf[0]:+.1%}, {rf[1]:+.1%}]")


def report() -> None:
    exo = exominer()
    s1 = json.loads((S1 / "preds-clues.json").read_text(encoding="utf-8"))
    lines = (DATA / "finetune" / "sft_v10" / "train.jsonl").read_text(encoding="utf-8").splitlines()
    trained = {json.loads(line)["toi"] for line in lines}
    rng = np.random.default_rng(SEED)
    rows, skipped = [], 0
    for split in SAVED:
        for s in val_signals(split):
            if s["toi"] not in exo or s["toi"] not in s1:
                continue
            t = tier(s, split)
            if t is None:
                skipped += 1
                continue
            rows.append({"toi": s["toi"], "split": split, "trained": s["toi"] in trained,
                         "planet": s["d"]["label"]["target"] == "gate2", "tier": t, "tessa": t == 2, "s1": s1[s["toi"]],
                         "score": t + 0.999 * s1[s["toi"]], "exo": exo[s["toi"]] >= 0.5, "exo_score": exo[s["toi"]],
                         "xgb": (s["d"]["checks"]["system1"]["planet_rank"] or 0) >= gate2.PASS_RANK})
    print(f"{len(rows)} shared TOIs, {skipped} with no model run left out")
    table("TEST (clean)", [r for r in rows if r["split"] == "test"], rng)
    table("VAL (tuned the floor)", [r for r in rows if r["split"] == "val"], rng)
    table("TRAIN, not in the training rows", [r for r in rows if r["split"] == "train" and not r["trained"]], rng)
    table("TRAIN, in the training rows (Glimmer learned these)", [r for r in rows if r["trained"]], rng)
    table("ALL (contaminated)", rows, rng)
    (DATA / "finetune" / "exo_bench.json").write_text(json.dumps(rows), encoding="utf-8")


def ffi() -> None:
    """The same rows against ExoMiner++'s full-frame run alone, the kind of data Tessa uses. Run report first."""
    with open(DATA / "finetune" / "exominer" / "matched_FFI.csv", encoding="utf-8") as f:
        exo = {r["toi"]: float(r["score"]) for r in csv.DictReader(f)}
    rng = np.random.default_rng(SEED)
    rows = json.loads((DATA / "finetune" / "exo_bench.json").read_text(encoding="utf-8"))
    rows = [r | {"exo_score": exo[r["toi"]], "exo": exo[r["toi"]] >= 0.5} for r in rows if r["toi"] in exo]
    print(f"{len(rows)} shared TOIs in ExoMiner++'s full-frame run")
    table("TEST (clean)", [r for r in rows if r["split"] == "test"], rng)
    table("VAL (tuned the floor)", [r for r in rows if r["split"] == "val"], rng)
    table("TRAIN, not in the training rows", [r for r in rows if r["split"] == "train" and not r["trained"]], rng)
    table("TRAIN, in the training rows (Glimmer learned these)", [r for r in rows if r["trained"]], rng)
    table("ALL (contaminated)", rows, rng)
    for name, rs in (("test", [r for r in rows if r["split"] == "test"]), ("all", rows)):
        y = np.array([r["planet"] for r in rs], int)
        a, b = np.array([r["s1"] for r in rs]), np.array([r["exo_score"] for r in rs])
        lo, hi = auc_ci(y, a, b, rng)
        print(f"System 1 minus ExoMiner++ FFI, {name}: AUC {roc_auc_score(y, a) - roc_auc_score(y, b):+.3f} [{lo:+.3f}, {hi:+.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", nargs="?", default="report", choices=["report", "run", "ffi"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    run(a.jobs) if a.stage == "run" else ffi() if a.stage == "ffi" else report()


if __name__ == "__main__":
    main()
