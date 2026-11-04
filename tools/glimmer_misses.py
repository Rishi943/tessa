"""The planets XGBoost (System 1, at its own 0.32 cut) would throw away on the held-out test split, and what happened to
each in Tessa: Glimmer passed it, or why not. No model runs: read from the saved test verdicts.

  python tools/glimmer_misses.py            the breakdown
  python tools/glimmer_misses.py --list     also one line per missed planet: its stop, rank, depth, SNR and Glimmer's reason

Descriptive only. The test split is not used to change Tessa: a fix for what it shows would be designed and checked on
val, then test scored once more.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from finetune.exo_bench import SAVED, tier, verdict  # noqa: E402
from finetune.valgate import val_signals  # noqa: E402
from pipeline import gate2  # noqa: E402


def main() -> None:
    rows = []
    for s in val_signals("test"):
        t = tier(s, "test")
        if t is None:
            continue
        c = s["d"]["checks"]
        ev = s["d"].get("evidence")
        rank = float((c.get("system1") or {}).get("planet_rank") or 0.0)
        hard = any(r.startswith(gate2.HARD_ROWS) for r in gate2.table_rows(c, ev))
        xgb = rank >= gate2.PASS_RANK and s["reaches"] and not hard
        planet = s["d"]["label"]["target"] == "gate2"
        why, v, d = "", None, {}
        if not s["reaches"]:
            why = "code rejected it before Glimmer: " + "; ".join(c.get("reasons", [])[:1])
        else:
            f = SAVED["test"] / f"{s['toi']}.json"
            text = json.loads(f.read_text(encoding="utf-8"))["text"]
            v = verdict(text)
            if t == 1 and v == "gate2":
                why = "Glimmer passed it; a hard row or the rank floor stopped it"
            elif v == "flag":
                why = "Glimmer flagged it"
            elif v == "reject":
                why = "Glimmer rejected it"
            elif v is None:
                why = "Glimmer's reply could not be read"
            try:
                d = json.loads(text[text.index("{"):text.rindex("}") + 1])
            except ValueError:
                d = {}
        sig = c.get("signal", {})
        rows.append({"toi": s["toi"], "planet": planet, "xgb": xgb, "glimmer": t == 2, "why": why, "rank": rank,
                     "depth": sig.get("depth_ppm"), "snr": sig.get("snr"), "dur": sig.get("dur_h"), "period": sig.get("period"),
                     "reason": (d.get("summary") or d.get("why") or "")[:220],
                     "checks": [r.get("check") for r in d.get("reasons", []) if isinstance(r, dict)][:4]})

    pool = [r for r in rows if not r["xgb"]]
    planets = [r for r in pool if r["planet"]]
    saved = [r for r in planets if r["glimmer"]]
    missed = [r for r in planets if not r["glimmer"]]
    fps = [r for r in pool if not r["planet"]]
    print(f"XGBoost at {gate2.PASS_RANK} in the pipeline would discard {len(pool)} of {len(rows)} test signals: "
          f"{len(planets)} planets, {len(fps)} FPs ({len(planets) / len(pool):.0%} planets)")
    print(f"Glimmer passed {len(saved) + sum(r['glimmer'] for r in fps)} of them: {len(saved)} planets, "
          f"{sum(r['glimmer'] for r in fps)} FPs")
    print(f"\nthe {len(missed)} planets in that pool that Tessa did not pass, by where they stopped:")
    for k, n in Counter(r["why"].split(":")[0] for r in missed).most_common():
        print(f"  {n:3}  {k}")
    print("\n  code's first reason, for those it stopped before Glimmer:")
    for k, n in Counter(r["why"].split(": ", 1)[1].split(" (")[0][:70] for r in missed if r["why"].startswith("code")).most_common(8):
        print(f"  {n:3}  {k}")
    print("\n  checks Glimmer cited first when it rejected or flagged one:")
    for k, n in Counter((r["checks"] or ["(none)"])[0] for r in missed if r["why"].startswith("Glimmer")).most_common(10):
        print(f"  {n:3}  {k}")
    med = lambda xs: sorted(xs)[len(xs) // 2] if xs else float("nan")
    for name, grp in (("saved", saved), ("missed", missed)):
        print(f"\n  {name:6}: median rank {med([r['rank'] for r in grp]):.3f}, depth {med([r['depth'] or 0 for r in grp]):.0f} ppm, "
              f"SNR {med([r['snr'] or 0 for r in grp]):.1f}, duration {med([r['dur'] or 0 for r in grp]):.1f} h, "
              f"period {med([r['period'] or 0 for r in grp]):.2f} d")
    if "--list" in sys.argv:
        print()
        for r in sorted(missed, key=lambda r: r["why"]):
            print(f"TOI {r['toi']}: {r['why'][:80]} | rank {r['rank']:.3f}, {r['depth'] or 0:.0f} ppm, SNR {r['snr'] or 0:.1f}"
                  f" | {r['checks']} | {r['reason']}")


if __name__ == "__main__":
    main()
