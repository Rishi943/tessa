"""Is TRICERATOPS's false-positive probability worth adding to System 1? A test on a sample of labelled signals.

The sample: train + val signals (test untouched) that are past the code rejections and that the planet-range
table flags (pipeline.gate2.outside_planet_range with the record's evidence): 40 planets (target gate2), 40 FPs
(reject) and 20 APCs (flag). One signal per star, and the ranks spread over each group's range: its signals sorted by
System 1's planet rank, cut into equal-count bins, one drawn from each (fixed seed). Drawn on 2026-10-30 with
gate 2's v8 table (rows fire at any rank); gate2.py then moved to v9, where a rank under PASS_RANK flags too, so
sample.json holds the draw and a new `sample` follows whatever outside_planet_range does by then.

What TRICERATOPS gets is what a new survey signal has: the star's TIC id, our own refit of the signal (period,
epoch, duration from the shape fit, box depth) and our own FFI light curve folded on it. No TOI name or disposition,
no ExoFOP data, no contrast curve; the TESS-SPOC apertures, TESScut cutouts, TIC and Gaia come from the public
archives inside the job. The folding is done in the job too (the light curves are the fetch job's output, which
stays on Kaggle), with 1 run per signal instead of 5: the runs of one signal differ by about 0.003 in FPP.

Run: python -m finetune.fpp_test sample [--seed 0]             draw the sample into data/finetune/fpp_test/sample.json (once)
     python -m finetune.fpp_test job JOB_DIR [--cut N]          the Kaggle job folder, for the first N signals of the sample
     python E:/AI/tools/kaggle-run/kaggle_run.py push JOB_DIR fpp_job.py SLUG --kernel tessa-ft-fetch
            --args="--jobs fpp_jobs.json --runs 1 [--limit 3] [--part I --parts N]"      (CPU, no --gpu)
     python E:/AI/tools/kaggle-run/kaggle_run.py fetch SLUG data/finetune/fpp_test/kaggle_SLUG
     python -m finetune.fpp_test score                           coverage, and AUC of FPP, rank and both
"""
import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from pipeline.gate2 import outside_planet_range
from pipeline.paths import DATA, ROOT

FOLDER = DATA / "finetune" / "fpp_test"
QUOTA = {"gate2": 40, "reject": 40, "flag": 20}
KIND = {"gate2": "planet", "reject": "FP", "flag": "APC"}
ENTRY = "from pipeline.fpp import main\n\nif __name__ == \"__main__\":\n    main()\n"
REPEATS = 20  # grouped cross-validation, reshuffled
BOOTSTRAPS = 1000


def eligible() -> dict[str, list[dict]]:
    """Train + val vet_s1 records past the code rejections that the table flags, by target (gate2, reject, flag)."""
    pool = defaultdict(list)
    for path in sorted((DATA / "finetune" / "vet_s1").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        c = d["checks"]
        if d["label"]["split"] not in ("train", "val") or c["code_verdict"] == "reject":
            continue
        rows = outside_planet_range(c, d.get("evidence"))
        if rows:
            pool[d["label"]["target"]].append({**d, "rank": (c.get("system1") or {}).get("planet_rank") or 0,
                                               "rows": sorted({r.split(":")[0] for r in rows})})
    return pool


def spread(n: int) -> list[int]:
    """0..n-1 in an order whose every prefix is spread evenly over the range (van der Corput): the job runs the
    signals in this order, so a time cap or --cut keeps the whole rank range."""
    def radical_inverse(i: int) -> float:
        x, f = 0.0, 0.5
        while i:
            x, i, f = x + f * (i & 1), i >> 1, f / 2
        return x
    return sorted(range(n), key=radical_inverse)


def job_order(signals: list[dict]) -> list[dict]:
    """The sample in job order: each group by `spread` of its rank bins, the groups interleaved in proportion."""
    pos = {k: spread(q) for k, q in ((KIND[t], q) for t, q in QUOTA.items())}
    key = lambda s: ((pos[s["kind"]].index(s["bin"]) + 0.5) / len(pos[s["kind"]]), s["kind"])  # noqa: E731
    return sorted(signals, key=key)


def draw(pool: dict[str, list[dict]], seed: int) -> list[dict]:
    """QUOTA signals per target, one per star, spread over the ranks (equal-count bins of System 1's rank)."""
    rng = random.Random(seed)
    taken, out = set(), []
    for target, quota in QUOTA.items():
        by_star = defaultdict(list)
        for d in pool[target]:
            if d["tic"] not in taken:
                by_star[d["tic"]].append(d)
        one = sorted((rng.choice(by_star[t]) for t in sorted(by_star)), key=lambda d: (d["rank"], d["toi"]))
        for i, bin_ in enumerate(np.array_split(np.arange(len(one)), quota)):
            d = one[int(rng.choice(list(bin_)))]
            taken.add(d["tic"])
            c = d["checks"]
            out.append({"toi": d["toi"], "tic": d["tic"], "kind": KIND[target], "disp": d["label"]["disp"],
                        "split": d["label"]["split"], "rank": d["rank"], "rows": d["rows"],
                        "bin": i,
                        "period": c["signal"]["period"], "t0": c["signal"]["t0"],
                        "t14_h": c["shape"]["t14_h"] or c["signal"]["dur_h"], "depth_ppm": c["signal"]["depth_ppm"]})
    return job_order(out)


def sample(seed: int) -> None:
    path = FOLDER / "sample.json"
    if path.exists():
        raise SystemExit(f"{path} exists: the draw is kept so every run scores the same signals; delete it to draw again")
    signals = draw(eligible(), seed)
    FOLDER.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "quota": QUOTA, "cut": None, "signals": signals}, indent=1), encoding="utf-8")
    print(f"{len(signals)} signals ({Counter(s['kind'] for s in signals)}), {len({s['tic'] for s in signals})} stars in {path}")


def job(job_dir: Path, cut: int | None) -> None:
    """The job folder for the first `cut` signals in job order; score then counts those."""
    path = FOLDER / "sample.json"
    drawn = json.loads(path.read_text(encoding="utf-8"))
    signals = job_order(drawn["signals"])[:cut]
    path.write_text(json.dumps({**drawn, "cut": cut, "signals": job_order(drawn["signals"])}, indent=1), encoding="utf-8")
    (job_dir / "pipeline").mkdir(parents=True, exist_ok=True)
    (job_dir / "finetune").mkdir(exist_ok=True)
    for name in ("__init__.py", "fpp.py", "search.py"):  # the job and its detrending
        shutil.copy(ROOT / "pipeline" / name, job_dir / "pipeline" / name)
    for name in ("__init__.py", "inputs.py"):  # shards_by_tic
        shutil.copy(ROOT / "finetune" / name, job_dir / "finetune" / name)
    # only what a new signal has: no name, label, rank or split
    jobs = [{k: s[k] for k in ("tic", "period", "t0", "t14_h", "depth_ppm")} for s in signals]
    (job_dir / "fpp_jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    (job_dir / "fpp_job.py").write_text(ENTRY, encoding="utf-8")
    (job_dir / "requirements.txt").write_text("triceratops==1.1.0\n", encoding="utf-8")
    print(f"{len(signals)} signals ({Counter(s['kind'] for s in signals)}), {len({s['tic'] for s in signals})} stars; "
          f"job folder {job_dir}")


def results(folder: Path) -> dict[int, dict]:
    """fpp_TIC.json under the folder; an FPP beats an uncovered or error file for the same star."""
    out = {}
    for p in folder.rglob("fpp_[0-9]*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["tic"] not in out or r.get("FPP_mean") is not None:
            out[r["tic"]] = r
    return out


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def stacked(X: np.ndarray, y: np.ndarray, hosts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold logistic scores averaged over reshuffled grouped folds, and each repeat's AUC."""
    oof, aucs = np.zeros((REPEATS, len(y))), []
    for r in range(REPEATS):
        for fit, held in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, y, hosts):
            oof[r, held] = LogisticRegression(C=1.0, max_iter=2000).fit(X[fit], y[fit]).decision_function(X[held])
        aucs.append(roc_auc_score(y, oof[r]))
    return oof.mean(0), np.array(aucs)


def score(folder: Path) -> None:
    drawn = json.loads((FOLDER / "sample.json").read_text(encoding="utf-8"))
    signals = drawn["signals"][:drawn["cut"]]
    res = results(folder)
    have = [s for s in signals if res.get(s["tic"], {}).get("FPP_mean") is not None]
    kinds = ("planet", "FP", "APC")
    print(f"covered {len(have)} of {len(signals)} signals: " + ", ".join(
        f"{k} {sum(s['kind'] == k for s in have)}/{sum(s['kind'] == k for s in signals)}" for k in kinds))
    why = Counter(re_why(res.get(s["tic"])) for s in signals if s not in have)
    for reason, n in why.most_common():
        print(f"  not covered ({n}): {reason}")
    aps = Counter(res[s["tic"]]["aperture"] for s in have)
    print(f"  apertures: {dict(aps)}; seconds per covered signal: median "
          f"{np.median([res[s['tic']]['seconds'] for s in have]):.0f}" if have else "  nothing covered")
    for k in kinds:
        v = [res[s["tic"]]["FPP_mean"] for s in have if s["kind"] == k]
        if v:
            print(f"  FPP of {k}s: median {np.median(v):.3f}, quartiles {np.percentile(v, 25):.3f}-{np.percentile(v, 75):.3f}")
    rs = [s for s in have if s["kind"] != "APC"]
    y = np.array([s["kind"] == "planet" for s in rs], int)
    if min(y.sum(), (1 - y).sum()) < 8:
        return print("planets vs FPs: too few covered to score")
    fpp = np.array([res[s["tic"]]["FPP_mean"] for s in rs])
    rank = np.array([s["rank"] for s in rs])
    hosts = np.array([s["tic"] for s in rs])
    X = np.column_stack([logit(rank), logit(1 - fpp)])
    both, aucs = stacked(X, y, hosts)
    rng = np.random.default_rng(0)
    gains = []
    for _ in range(BOOTSTRAPS):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            gains.append(roc_auc_score(y[i], both[i]) - roc_auc_score(y[i], rank[i]))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    print(f"planets vs FPs ({y.sum()} / {(1 - y).sum()} signals, one per star), AUC (0.5 = coin flip, 1 = every planet above every FP):")
    print(f"  FPP alone      {roc_auc_score(y, -fpp):.3f}")
    print(f"  rank alone     {roc_auc_score(y, rank):.3f}")
    print(f"  rank + FPP     {roc_auc_score(y, both):.3f}  (grouped 5-fold CV, {REPEATS} reshuffles: per-reshuffle "
          f"{aucs.mean():.3f} +- {aucs.std():.3f}); gain over rank {roc_auc_score(y, both) - roc_auc_score(y, rank):+.3f}, "
          f"bootstrap 95% interval {lo:+.3f} to {hi:+.3f}")


def re_why(r: dict | None) -> str:
    """The reason a signal has no FPP, in a few words."""
    if r is None:
        return "never ran (job stopped at its time cap, or still running)"
    if "error" in r:
        return "error: " + r["error"][:100]
    return (r.get("uncovered") or "no FPP").split(":")[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample").add_argument("--seed", type=int, default=0)
    j = sub.add_parser("job")
    j.add_argument("job_dir", type=Path)
    j.add_argument("--cut", type=int, help="keep the first N signals in job order (a time budget)")
    c = sub.add_parser("score")
    c.add_argument("dir", type=Path, nargs="?", default=FOLDER, help="where the fetched kaggle_* folders are")
    a = ap.parse_args()
    if a.cmd == "sample":
        sample(a.seed)
    elif a.cmd == "job":
        job(a.job_dir, a.cut)
    else:
        score(a.dir)


if __name__ == "__main__":
    main()
