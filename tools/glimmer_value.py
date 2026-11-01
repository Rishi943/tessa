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
     python tools/glimmer_value.py funnel            the survey funnel recounted from the saved files (for the README)
     python tools/glimmer_value.py xgb-only OUT_DIR  the signals XGBoost alone would pass and Glimmer did not: gate 2 as if
                                                     XGBoost had passed them (novelty, test, seasons), and their folded light
                                                     curves in OUT_DIR for tools/fpp_survivors.py --folded OUT_DIR
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
    # Is Glimmer just a looser cut? The XGBoost cut that keeps every candidate, and how many signals it then passes.
    kept = [by[t]["rank"] for t in CANDIDATES if t in by]
    if kept:
        need = min(kept)
        n = sum(r["rank"] >= need for r in rows)
        print(f"to keep all {len(kept)} candidates XGBoost needs rank >= {need:.3f}: it then passes {n} of {len(rows)} "
              f"signals; Glimmer passed {len(gp)} and kept all {sum(by[t]['passes'] for t in CANDIDATES if t in by)}")
        for k in sorted(kept, reverse=True):
            print(f"  XGBoost >= {k:.3f}: passes {sum(r['rank'] >= k for r in rows):>3} signals, "
                  f"keeps {sum(x >= k for x in kept)} of {len(kept)} candidates")


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


def survey_rows() -> list[dict]:
    rows = []
    for r in json.loads((DATA / "survey_v10" / "runset.json").read_text(encoding="utf-8")):
        c = json.loads((DATA / "vet" / f"{r['batch']}_v10" / f"{r['tic']}.json").read_text(encoding="utf-8"))
        d = DATA / "dossiers" / f"{r['tic']}.glimmer-vetter-v10.v10.json"
        v = json.loads(d.read_text(encoding="utf-8"))["dossier"]["verdict"] if d.exists() else None
        rows.append({"tic": r["tic"], "batch": r["batch"], "c": c, "rank": rank(c), "glimmer": v,
                     "passes": v == "gate2" and not gate2.outside_planet_range(c, gate2.evidence(r["tic"]), "v10")})
    return rows


def funnel() -> None:
    """Every survey count the README states, from the saved files, so README and form quote one source."""
    import csv
    rows = survey_rows()
    v = [r["glimmer"] for r in rows]
    print(f"run set (reached Glimmer): {len(rows)}")
    print(f"Glimmer verdicts: gate2 {v.count('gate2')}, flag {v.count('flag')}, reject {v.count('reject')}, none {v.count(None)}")
    print(f"gate2 and no hard row / rank floor (a Tessa pass): {sum(r['passes'] for r in rows)}")
    total = {}
    for batch in ("survey", "batch2"):
        f = DATA / "vet" / f"{batch}_v10" / "status.csv"
        if not f.exists():
            print(f"{f} missing: run `python tools/survey_v10.py gate2`")
            continue
        with open(f, encoding="utf-8") as fh:
            st = [r["status"] for r in csv.DictReader(fh)]
        for s in set(st):
            total[s] = total.get(s, 0) + st.count(s)
        print(f"gate 2 status, {batch}_v10: " + ", ".join(f"{s} {st.count(s)}" for s in sorted(set(st))))
    print("gate 2 status, both: " + ", ".join(f"{s} {n}" for s, n in sorted(total.items())) + f"  (total {sum(total.values())})")
    cands = sorted(int(p.stem) for b in ("survey", "batch2") for p in (DATA / "vet" / f"{b}_v10" / "candidates").glob("*.json"))
    print(f"gate 2 candidates (before TRICERATOPS): {len(cands)}: {cands}")
    # TRICERATOPS on the v10 passes: every fpp_out* run under data/survey_v10, later folders win for a star run twice
    fpp_rows = {}
    for res in sorted((DATA / "survey_v10").glob("fpp_out*/results.json")):
        for r in json.loads(res.read_text(encoding="utf-8")):
            if r.get("ran") or r["tic"] not in fpp_rows:
                fpp_rows[r["tic"]] = r | {"from": res.parent.name}
    if not fpp_rows:
        print("no TRICERATOPS results under data/survey_v10/fpp_out*")
        return
    print(f"\nTRICERATOPS (survey bar FPP < 0.5 and NFPP < 0.1; likely planet NFPP < 0.001):")
    passed = []
    for tic in cands:
        r = fpp_rows.get(tic)
        if r is None or not r.get("ran"):
            print(f"  {tic:>10}  no TRICERATOPS result" + (f" ({r.get('why')})" if r else ""))
            continue
        ok = r["FPP"] < 0.5 and r["NFPP"] < 0.1
        tier_ = "likely planet" if ok and r["NFPP"] < 0.001 else "nearby star to rule out" if ok else "FAILS the survey bar"
        passed += [tic] if ok else []
        print(f"  {tic:>10}  FPP {r['FPP']:.4f}  NFPP {r['NFPP']:.2e}  {r['class']:28} {tier_}  [{r['from']}]")
    print(f"pass TRICERATOPS: {len(passed)} of {len(cands)}: {passed}")
    others = sorted(t for t in fpp_rows if t not in cands)
    if others:
        print(f"also run through TRICERATOPS but not gate 2 candidates now: {others}")
    ran = [r for r in fpp_rows.values() if r.get("ran")]
    ok = [r["tic"] for r in ran if r["FPP"] < 0.5 and r["NFPP"] < 0.1]
    print(f"\nall TRICERATOPS runs: {len(fpp_rows)} stars, {len(ran)} ran, {len(ok)} pass the survey bar")
    # stars the README, form or old site name: their TRICERATOPS result and current gate 2 status with its reason
    status = {}
    for batch in ("survey", "batch2"):
        f = DATA / "vet" / f"{batch}_v10" / "status.csv"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                status |= {int(r["tic"]): (r["status"], r["reasons"]) for r in csv.DictReader(fh)}
    print("named stars:")
    for tic in (50099421, 240488148, 316447391, 426450210):
        r = fpp_rows.get(tic) or {}
        tri = (f"FPP {r['FPP']:.4f} NFPP {r['NFPP']:.2e} {r['class']}" if r.get("ran") else "no TRICERATOPS result")
        st, why = status.get(tic, ("not in status.csv", ""))
        print(f"  {tic:>10}  {tri:52}  gate 2: {st}" + (f" ({why[:150]})" if why else ""))


def xgb_only(out: Path) -> None:
    """Signals XGBoost at its v9 cut passes and Glimmer did not: what gate 2 says if XGBoost had passed them, and their folds."""
    from pipeline import fpp
    from pipeline.paths import CACHE
    from pipeline.search import detrend
    from pipeline.vet import load_lc, signal_from_results
    lc_dir = {"survey": CACHE / "lc", "batch2": CACHE / "lc_batch2"}
    results = {"survey": DATA / "survey_results.csv", "batch2": DATA / "batch2_results.csv"}
    picks = [r for r in survey_rows() if r["rank"] >= gate2.PASS_RANK and not r["passes"]]
    print(f"{len(picks)} signals XGBoost at {gate2.PASS_RANK} passes and Glimmer did not")
    out.mkdir(parents=True, exist_ok=True)
    for r in picks:
        tic, c = r["tic"], r["c"]
        hard_rows = gate2.outside_planet_range(c, gate2.evidence(tic), "v10")
        as_pass = {"prompt": "v10", "dossier": {"verdict": "gate2", "reasons": [], "public_summary": ""}}
        sig = signal_from_results(tic, results[r["batch"]])
        status, reasons, _ = gate2.decide(c, as_pass, sig, lc_dir[r["batch"]])
        print(f"TIC {tic} ({r['batch']}): XGBoost rank {r['rank']:.3f}, Glimmer said {r['glimmer']}; "
              f"hard rows {hard_rows or 'none'}; gate 2 if passed: {status}" + (f" ({'; '.join(reasons)[:160]})" if reasons else ""))
        if status != "candidate":
            continue
        s = c["signal"]
        raw = load_lc(tic, lc_dir[r["batch"]])
        lc = detrend(raw["time"], raw["flux"], raw["sector"])
        try:
            f = fpp.fold(tic, s["period"], s["t0"], s["dur_h"], s["depth_ppm"], raw, lc)
        except fpp.Uncovered as e:
            print(f"  TRICERATOPS cannot cover it: {e}")
            continue
        (out / f"fpp_input_{tic}.json").write_text(json.dumps(f), encoding="utf-8")
        print(f"  folded -> {out / f'fpp_input_{tic}.json'}")
    print(f"\nnext, in the TRICERATOPS venv: python tools/fpp_survivors.py --folded {out} --out {out / 'fpp'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "funnel":
        funnel()
        sys.exit()
    if len(sys.argv) > 2 and sys.argv[1] == "xgb-only":
        xgb_only(Path(sys.argv[2]))
        sys.exit()
    for part in (part1, part2, part3):
        try:
            part()
        except FileNotFoundError as e:
            print(f"  skipped, missing file: {e.filename}")
