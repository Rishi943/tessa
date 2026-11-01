"""Triage: System 1 settles the clear cases, Glimmer investigates only the uncertain band. Dev picks the cuts, test runs once.

Reads finetune/exo_bench.json (no model runs; Glimmer v10's saved verdicts). For a TOI with System 1 score s:
  s <  lo        rejected on the score alone
  s >= hi        passed on the score alone
  lo <= s < hi   Glimmer decides: `tessa` (Glimmer says gate2 and the code's hard rows and rank floor agree) or `tier` (pass or flag)
Setting hi above 1 gives the veto design (Glimmer must agree everywhere above lo); lo == hi gives System 1 alone.
Dev = val + train signals Glimmer did not train on. The cuts are the ones that recover at least as many planets as ExoMiner++ at
0.5 with the fewest false positives; ties go to the narrower Glimmer band. Test is then scored once, with paired bootstrap intervals.

Run: python -m finetune.triage
"""
import argparse
import itertools

import numpy as np
from sklearn.metrics import roc_auc_score

from finetune.blend import SEED, load, rates, y
from finetune.exo_bench import ci


def decide(rows: list[dict], lo: float, hi: float, how: str) -> np.ndarray:
    s = np.array([r["s1"] for r in rows])
    g = np.array([r["tessa"] if how == "tessa" else r["tier"] >= 1 for r in rows])
    return (s >= hi) | ((s >= lo) & (s < hi) & g)


def choose(devs: list[dict], how: str, want: int, veto_only: bool = False):
    t = y(devs)
    s = np.array([r["s1"] for r in devs])
    cuts = np.unique(s)
    best = None
    for lo in cuts:
        for hi in ([np.inf] if veto_only else [h for h in cuts if h >= lo]):
            p = decide(devs, lo, hi, how)
            a, b = rates(p, t)
            if a < want:
                continue
            width = int(((s >= lo) & (s < hi)).sum())
            key = (b, width, -lo)
            if best is None or key < best[0]:
                best = (key, float(lo), float(hi), a, b, width)
    return best


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    rows = load()
    devs = [r for r in rows if r["split"] == "val" or (r["split"] == "train" and not r["trained"])]
    tests = [r for r in rows if r["split"] == "test"]
    td, tt = y(devs), y(tests)
    want = rates(np.array([r["exo"] for r in devs]), td)[0]
    print(f"dev: {len(devs)} TOIs ({td.sum()} planets, {(1 - td).sum()} FPs); ExoMiner++ recovers {want} planets with "
          f"{rates(np.array([r['exo'] for r in devs]), td)[1]} FPs")
    print(f"test: {len(tests)} TOIs ({tt.sum()} planets, {(1 - tt).sum()} FPs)\n")

    designs = []
    s1d = np.array([r["s1"] for r in devs])
    cut = max(c for c in np.unique(s1d) if rates(s1d >= c, td)[0] >= want)
    designs.append(("System 1 alone", cut, cut, "tessa"))
    for how in ("tessa", "tier"):
        for name, veto in ((f"veto ({how})", True), (f"triage ({how})", False)):
            b = choose(devs, how, want, veto)
            if b is None:
                top = rates(decide(devs, -1.0, np.inf, how), td)[0]
                print(f"dev pick {name:16} cannot reach {want} planets: even with no score cut it passes only {top}")
                continue
            designs.append((name, b[1], b[2], how))
            print(f"dev pick {name:16} lo {b[1]:.4f} hi {b[2]:.4f}: {b[3]} planets, {b[4]} FPs, Glimmer sees {b[5]}/{len(devs)}")
    print()

    rng = np.random.default_rng(SEED)
    em = np.array([r["exo_score"] >= 0.5 for r in tests], float)
    a, b = rates(em > 0, tt)
    print(f"{'test':18} {'planets':>12} {'FPs':>14} {'Glimmer sees':>13}  vs ExoMiner++ planets / FPs (95% CI)")
    print(f"{'ExoMiner++ at 0.5':18} {a:>5}/{tt.sum()} {a / tt.sum():4.0%} {b:>5}/{(1 - tt).sum()} {b / (1 - tt).sum():5.1%}")
    for name, lo, hi, how in designs:
        p = decide(tests, lo, hi, how)
        a, b = rates(p, tt)
        seen = int(((np.array([r['s1'] for r in tests]) >= lo) & (np.array([r['s1'] for r in tests]) < hi)).sum())
        pf = p.astype(float)
        rp, rf = ci(pf[tt == 1], em[tt == 1], rng), ci(pf[tt == 0], em[tt == 0], rng)
        print(f"{name:18} {a:>5}/{tt.sum()} {a / tt.sum():4.0%} {b:>5}/{(1 - tt).sum()} {b / (1 - tt).sum():5.1%} {seen:>8}/{len(tests)}  "
              f"{pf[tt == 1].mean() - em[tt == 1].mean():+.1%} [{rp[0]:+.1%}, {rp[1]:+.1%}] / "
              f"{pf[tt == 0].mean() - em[tt == 0].mean():+.1%} [{rf[0]:+.1%}, {rf[1]:+.1%}]")
    # Fixed bands: Glimmer sees the TOIs just below the System 1 cut (rescue) and just above it (check); widths are shares of dev
    # TOIs, set from dev only, nothing is tuned. Recall is not matched here, so read planets and FPs together.
    print("\nfixed band around the System 1 cut (Glimmer = tessa), edges from dev ranks:")
    print(f"{'band':>6} {'dev planets/FPs':>16} {'test planets':>13} {'test FPs':>10} {'Glimmer sees':>13}")
    ranked = np.sort(s1d)
    pos = int(np.searchsorted(ranked, cut))
    s1t = np.array([r["s1"] for r in tests])
    for share in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
        half = int(round(share * len(devs) / 2))
        lo = ranked[max(pos - half, 0)]
        hi = ranked[min(pos + half, len(ranked) - 1)] if share else cut
        da, db = rates(decide(devs, lo, hi, "tessa"), td)
        ta, tb = rates(decide(tests, lo, hi, "tessa"), tt)
        print(f"{share:>6.0%} {da:>7}/{db:<8} {ta:>6}/{tt.sum():<6} {tb:>5}/{(1 - tt).sum():<4} {int(((s1t >= lo) & (s1t < hi)).sum()):>8}/{len(tests)}")
    print(f"\ntest AUC: System 1 {roc_auc_score(tt, [r['s1'] for r in tests]):.3f}, ExoMiner++ {roc_auc_score(tt, [r['exo_score'] for r in tests]):.3f}")


if __name__ == "__main__":
    main()
