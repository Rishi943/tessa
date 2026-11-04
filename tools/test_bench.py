"""The held-out test split in one small file, so its headline tables can be recomputed from a clean clone.

  python tools/test_bench.py export     writes runs/benchmarks/test_split.json from the saved test verdicts in data/
  python tools/test_bench.py            reads that file and prints the tables the README quotes

One row per test signal: its known outcome, System 1's rank, whether code let it reach Glimmer, whether a hard row
binds it, and the call of Glimmer v10 and of the base Muse Glimmer 30B (same server, same prompt v10, same input):
2 a Tessa pass, 1 stopped short (flag, or a pass a hard row or the rank floor stops), 0 rejected. The raw replies
(42 MB) stay in data/finetune/testgate and testgate_glimmer-base.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
OUT = ROOT / "runs" / "benchmarks" / "test_split.json"
XGB_CUT = 0.32  # pipeline/gate2.py PASS_RANK, the cut XGBoost used on its own in vetting v9


def export() -> None:
    import glimmer_value as gv
    import numpy as np
    from finetune.exo_bench import SAVED, verdict
    from finetune.valgate import val_signals
    from pipeline import gate2
    assert gate2.PASS_RANK == XGB_CUT

    def call(s: dict, folder: Path) -> int | None:
        if not s["reaches"]:
            return 0
        f = folder / f"{s['toi']}.json"
        if not f.exists():
            return None
        v = verdict(json.loads(f.read_text(encoding="utf-8"))["text"])
        if v in ("reject", None):
            return 0
        return 2 if v == "gate2" and not gate2.outside_planet_range(s["d"]["checks"], s["d"].get("evidence"), "v10") else 1

    rows = []
    for s in val_signals("test"):
        c = s["d"]["checks"]
        v10, base = call(s, SAVED["test"]), call(s, SAVED["test"].with_name("testgate_glimmer-base"))
        if v10 is None or base is None:
            continue
        rows.append({"toi": s["toi"], "planet": s["d"]["label"]["target"] == "gate2",
                     "rank": round(float((c.get("system1") or {}).get("planet_rank") or 0.0), 4),
                     "reaches": bool(s["reaches"]),
                     "hard": any(r.startswith(gate2.HARD_ROWS) for r in gate2.table_rows(c, s["d"].get("evidence"))),
                     "glimmer_v10": v10, "glimmer_base": base})
    val, _ = gv.labelled("val")
    yv = np.array([r["planet"] for r in val])
    matched = gv.matched_cut(val, np.array([r["glimmer"] for r in val])[yv].mean())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"xgb_cut": XGB_CUT, "xgb_cut_matched_on_val": round(float(matched), 4), "rows": rows},
                              separators=(",", ":")), encoding="utf-8")
    print(f"{len(rows)} test signals -> {OUT.relative_to(ROOT)}")


def report() -> None:
    d = json.loads(OUT.read_text(encoding="utf-8"))
    rows = d["rows"]
    P, F = sum(r["planet"] for r in rows), sum(not r["planet"] for r in rows)
    xgb = lambda r, cut: r["reaches"] and not r["hard"] and r["rank"] >= cut

    def line(name: str, sel) -> None:
        p = sum(1 for r in rows if sel(r) and r["planet"])
        f = sum(1 for r in rows if sel(r) and not r["planet"])
        print(f"  {name:52s} {p:4d} ({p / P:.0%})  {f:4d} ({f / F:.1%})  {p / max(p + f, 1):.0%}")

    print(f"test split: {P} planets, {F} false positives\n  {'who makes the call':52s} planets passed  FPs passed  precision")
    line("Glimmer v10 (fine-tuned)", lambda r: r["glimmer_v10"] == 2)
    line("Muse Glimmer 30B, base", lambda r: r["glimmer_base"] == 2)
    line(f"XGBoost, cut matched on val ({d['xgb_cut_matched_on_val']})", lambda r: xgb(r, d["xgb_cut_matched_on_val"]))
    line(f"XGBoost at its v9 cut ({d['xgb_cut']})", lambda r: xgb(r, d["xgb_cut"]))
    pool = [r for r in rows if not xgb(r, d["xgb_cut"])]
    print(f"\nXGBoost at {d['xgb_cut']} discards {len(pool)} signals, {sum(r['planet'] for r in pool)} of them planets. Passed again:")
    for key in ("glimmer_v10", "glimmer_base"):
        p = sum(1 for r in pool if r[key] == 2 and r["planet"])
        f = sum(1 for r in pool if r[key] == 2 and not r["planet"])
        print(f"  {key:12s} {p} planets for {f} false positives ({p / max(p + f, 1):.0%} planets)")


if __name__ == "__main__":
    export() if sys.argv[1:] == ["export"] else report()
