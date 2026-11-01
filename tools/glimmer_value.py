"""What does Glimmer add over XGBoost (System 1)? Three questions, no model runs: everything is read from saved verdicts.

1. Matched recall on the full test split. Tessa (Glimmer v10 + hard rows + rank floor) against the same pipeline with an
   XGBoost rank cut in Glimmer's place: if the cut is set so XGBoost passes as many planets as Glimmer, how many false
   positives does it let through? Two cuts: matched on test itself (descriptive), and matched on val, then applied to test.
2. The survey. For the six candidates and every signal Glimmer passed: System 1's rank, and whether XGBoost at its v9
   cut (PASS_RANK) would have passed it. Also how far the two agree across the run set.
3. Triage on the ExoMiner++-shared TOIs (finetune/exo_bench.json): System 1 settles the clear cases, Glimmer the band
   between two cuts chosen on dev (val + train signals Glimmer did not train on), test scored once.

Needs data/ (vet files, testgate/valgate verdicts, survey_v10 dossiers, exo_bench.json from `python -m finetune.exo_bench`).
Run: python tools/glimmer_value.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.exo_bench import SAVED, SEED, ci, tier  # noqa: E402
from finetune.valgate import val_signals  # noqa: E402
from pipeline import gate2  # noqa: E402
from pipeline.paths import DATA  # noqa: E402

CANDIDATES = {71503196: "likely", 116046773: "likely", 166321785: "likely",
              321290256: "nearby open", 50099421: "nearby open", 61220826: "nearby open"}


def rank(c: dict) -> float:
    return float((c.get("system1") or {}).get("planet_rank") or 0.0)


def hard(c: dict, ev: dict | None) -> bool:
    return any(r.startswith(gate2.HARD_ROWS) for r in gate2.table_rows(c, ev))


def labelled(split: str) -> tuple[list[dict], int]:
    rows, skipped = [], 0
    for s in val_signals(split):
        t = tier(s, split)
        if t is None:
            skipped += 1
            continue
        c = s["d"]["checks"]
        rows.append({"planet": s["d"]["label"]["target"] == "gate2", "glimmer": t == 2, "reaches": s["reaches"],
                     "hard": hard(c, s["d"].get("evidence")), "rank": rank(c)})
    return rows, skipped


def xgb(rows: list[dict], cut: float, pipeline: bool = True) -> np.ndarray:
    """XGBoost in Glimmer's place: same reach and hard rows (pipeline), or the rank alone."""
    return np.array([r["rank"] >= cut and (not pipeline or (r["reaches"] and not r["hard"])) for r in rows])


def counts(p: np.ndarray, y: np.ndarray) -> tuple[int, int]:
    return int(p[y].sum()), int(p[~y].sum())


def matched_cut(rows: list[dict], want_share: float) -> float | None:
    """Highest rank cut at which the XGBoost pipeline passes at least want_share of the planets."""
    y = np.array([r["planet"] for r in rows])
    for cut in sorted({r["rank"] for r in rows}, reverse=True):
        if counts(xgb(rows, cut), y)[0] >= want_share * y.sum():
            return cut
    return None


def line(name: str, p: np.ndarray, y: np.ndarray) -> None:
    a, b = counts(p, y)
    prec = a / (a + b) if a + b else float("nan")
    print(f"  {name:44} planets {a:>3}/{y.sum()} ({a / y.sum():4.0%})  FPs {b:>3}/{(~y).sum()} ({b / (~y).sum():5.1%})  "
          f"precision {prec:4.0%}")


def part1() -> None:
    print("=" * 100, "\n1. MATCHED RECALL, full test split\n")
    test, skip_t = labelled("test")
    val, skip_v = labelled("val")
    y = np.array([r["planet"] for r in test])
    g = np.array([r["glimmer"] for r in test])
    print(f"test: {len(test)} signals ({y.sum()} planets, {(~y).sum()} FPs), {skip_t} with no saved Glimmer run left out")
    print("  (sanity check: the README's full-test row was 213 planets, 18 FPs, on 297 planets / 282 FPs)\n")
    line("Tessa: Glimmer v10 makes the call", g, y)
    line(f"XGBoost rank >= {gate2.PASS_RANK} alone (v9 cut, no pipeline)", xgb(test, gate2.PASS_RANK, False), y)
    line(f"XGBoost rank >= {gate2.PASS_RANK} in Glimmer's place", xgb(test, gate2.PASS_RANK), y)

    rng = np.random.default_rng(SEED)
    cut_t = matched_cut(test, g[y].mean())
    yv = np.array([r["planet"] for r in val])
    gv = np.array([r["glimmer"] for r in val])
    cut_v = matched_cut(val, gv[yv].mean())
    for name, cut in ((f"matched on test (descriptive)", cut_t), (f"matched on val ({skip_v} val skipped), applied", cut_v)):
        if cut is None:
            print(f"  XGBoost cut {name}: cannot reach Glimmer's recall even at rank 0 (the pipeline's reach caps it)")
            continue
        p = xgb(test, cut)
        line(f"XGBoost >= {cut:.3f}, {name}", p, y)
        lo, hi = ci(g[~y].astype(float), p[~y].astype(float), rng)
        print(f"      Glimmer minus XGBoost, FP rate: {g[~y].mean() - p[~y].mean():+.1%} [{lo:+.1%}, {hi:+.1%}] (95% paired bootstrap)")
    both = g & xgb(test, gate2.PASS_RANK)
    only_g, only_x = g & ~xgb(test, gate2.PASS_RANK), ~g & xgb(test, gate2.PASS_RANK)
    print(f"\n  disagreements against XGBoost at {gate2.PASS_RANK} in the pipeline: both pass {int(both.sum())}; "
          f"Glimmer only {int(only_g.sum())} ({int(only_g[y].sum())} planets, {int(only_g[~y].sum())} FPs); "
          f"XGBoost only {int(only_x.sum())} ({int(only_x[y].sum())} planets, {int(only_x[~y].sum())} FPs)")


def part2() -> None:
    print("\n" + "=" * 100, "\n2. THE SURVEY\n")
    run = json.loads((DATA / "survey_v10" / "runset.json").read_text(encoding="utf-8"))
    rows = []
    for r in run:
        c = json.loads((DATA / "vet" / f"{r['batch']}_v10" / f"{r['tic']}.json").read_text(encoding="utf-8"))
        d = DATA / "dossiers" / f"{r['tic']}.glimmer-vetter-v10.v10.json"
        v = json.loads(d.read_text(encoding="utf-8"))["dossier"]["verdict"] if d.exists() else None
        rows.append({"tic": r["tic"], "rank": rank(c), "s1_only": r["s1_only"], "glimmer": v,
                     "passes": v == "gate2" and not gate2.outside_planet_range(c, gate2.evidence(r["tic"]), "v10")})
    by = {r["tic"]: r for r in rows}
    print(f"{'TIC':>10} {'class':12} {'XGB rank':>9} {'XGBoost v9 cut':>15} {'code had rejected it on rank':>29}")
    for tic, cls in CANDIDATES.items():
        r = by.get(tic)
        if r is None:
            print(f"{tic:>10} {cls:12} not in the run set")
            continue
        print(f"{tic:>10} {cls:12} {r['rank']:>9.3f} {'pass' if r['rank'] >= gate2.PASS_RANK else 'REJECT':>15} "
              f"{'yes' if r['s1_only'] else 'no':>29}")
    missed = [t for t in CANDIDATES if t in by and by[t]["rank"] < gate2.PASS_RANK]
    print(f"\n{len(missed)} of {len(CANDIDATES)} candidates are below XGBoost's v9 cut: {missed}")
    gp = [r for r in rows if r["passes"]]
    xp = [r for r in rows if r["rank"] >= gate2.PASS_RANK]
    print(f"run set {len(rows)}: Glimmer passed {len(gp)} ({sum(r['rank'] < gate2.PASS_RANK for r in gp)} of them below the "
          f"v9 cut); XGBoost at {gate2.PASS_RANK} would pass {len(xp)}; both {sum(r['passes'] for r in xp)}; "
          f"no Glimmer verdict {sum(r['glimmer'] is None for r in rows)}")


def part3() -> None:
    print("\n" + "=" * 100, "\n3. TRIAGE on the ExoMiner++-shared TOIs\n")
    f = DATA / "finetune" / "exo_bench.json"
    if not f.exists():
        print("exo_bench.json missing: run `python -m finetune.exo_bench` first")
        return
    rows = json.loads(f.read_text(encoding="utf-8"))
    devs = [r for r in rows if r["split"] == "val" or (r["split"] == "train" and not r["trained"])]
    tests = [r for r in rows if r["split"] == "test"]

    def arr(rs, k):
        return np.array([r[k] for r in rs])

    def decide(rs, lo, hi):
        s, g = arr(rs, "s1"), arr(rs, "tessa").astype(bool)
        return (s >= hi) | ((s >= lo) & (s < hi) & g)

    yd, yt = arr(devs, "planet").astype(bool), arr(tests, "planet").astype(bool)
    want = counts(arr(devs, "exo").astype(bool), yd)[0]
    cuts = np.unique(arr(devs, "s1"))
    s1_cut = max(c for c in cuts if counts(arr(devs, "s1") >= c, yd)[0] >= want)
    best = None
    for lo in cuts:
        for hi in cuts[cuts >= lo]:
            a, b = counts(decide(devs, lo, hi), yd)
            if a >= want:
                width = int(((arr(devs, "s1") >= lo) & (arr(devs, "s1") < hi)).sum())
                key = (b, width, -lo)
                if best is None or key < best[0]:
                    best = (key, float(lo), float(hi))
    print(f"dev {len(devs)} TOIs; designs must recover at least {want} dev planets (ExoMiner++ at 0.5 on dev)")
    print(f"test {len(tests)} TOIs ({yt.sum()} planets, {(~yt).sum()} FPs)\n")
    line("ExoMiner++ at 0.5", arr(tests, "exo").astype(bool), yt)
    line("Tessa: Glimmer v10 makes the call", arr(tests, "tessa").astype(bool), yt)
    line(f"System 1 alone, cut {s1_cut:.3f} set on dev", arr(tests, "s1") >= s1_cut, yt)
    if best:
        _, lo, hi = best
        seen = int(((arr(tests, "s1") >= lo) & (arr(tests, "s1") < hi)).sum())
        line(f"Triage: Glimmer decides s1 in [{lo:.3f}, {hi:.3f})", decide(tests, lo, hi), yt)
        print(f"      Glimmer sees {seen}/{len(tests)} test TOIs; System 1 settles the rest")
    else:
        print("  triage: no band reaches the dev recall target")


if __name__ == "__main__":
    for part in (part1, part2, part3):
        try:
            part()
        except FileNotFoundError as e:
            print(f"  skipped, missing file: {e.filename}")
