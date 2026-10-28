"""Before/after for System 1 versions: did a change help, and where?

Each version is a `python -m finetune.system1 --name NAME` run (preds-NAME.json: train out-of-fold, val,
test). Scored the same way every time:
  - test split: AUC, and at the version's own 2% conformal cut (System 1 alone) planets lost / FPs caught
  - the TOIs NASA's ExoMiner++ 2.0 also scored blind (train out-of-fold + val + test): AUC next to ExoMiner++
  - where: test AUC by subset, and FPs caught by why follow-up rejected them
  - which test TOIs crossed the cut, each way

Run: python -m finetune.compare base multis
"""
import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from finetune.system1 import OUT, FOLDER, load

SUBSETS = {
    "agent-bound": lambda r: r["code"] == "agent",
    "code-rejected": lambda r: r["code"] == "reject",
    "K/M hosts": lambda r: (r["label"].get("teff") or 9999) < 5300,
    "alerted 2024+": lambda r: bool(r["label"].get("recent")),
}


def exominer() -> dict[str, float]:
    """ExoMiner++ 2.0 mean score per TOI, averaged over its 2-min and FFI sector runs."""
    import csv
    scores: dict[str, list[float]] = {}
    for kind in ("2-min", "FFI"):
        with open(FOLDER / "exominer" / f"matched_{kind}.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                scores.setdefault(row["toi"], []).append(float(row["score"]))
    return {t: float(np.mean(s)) for t, s in scores.items()}


def auc(rows, p):
    y = np.array([r["label"]["target"] == "gate2" for r in rows])
    return roc_auc_score(y, [p[r["toi"]] for r in rows]) if 0 < y.sum() < len(y) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("versions", nargs="+")
    args = ap.parse_args()
    rows = [r for r in load() if r["label"]["target"] in ("gate2", "reject")]
    test = [r for r in rows if r["label"]["split"] == "test"]
    preds = {v: json.loads((OUT / f"preds-{v}.json").read_text(encoding="utf-8")) for v in args.versions}
    cuts = {v: json.loads((OUT / f"results-{v}.json").read_text(encoding="utf-8"))["cuts"]["0.02"] for v in args.versions}
    exo = exominer()
    shared = [r for r in rows if r["toi"] in exo and all(r["toi"] in p for p in preds.values())]

    width = max(len(v) for v in args.versions + ["ExoMiner++"]) + 2
    print(f"{'':{width}} {'test AUC':>9} {'lost@2%':>8} {'caught@2%':>10} {'shared AUC':>11}   (shared: {len(shared)} TOIs ExoMiner++ also scored blind)")
    yt = np.array([r["label"]["target"] == "gate2" for r in test])
    for v in args.versions:
        drop = np.array([preds[v][r["toi"]] < cuts[v] for r in test])
        print(f"{v:{width}} {auc(test, preds[v]):9.3f} {drop[yt].mean():8.1%} {drop[~yt].mean():10.1%} {auc(shared, preds[v]):11.3f}")
    print(f"{'ExoMiner++':{width}} {'':>9} {'':>8} {'':>10} {auc(shared, exo):11.3f}")

    print("\nwhere (test AUC by subset):")
    for name, keep in SUBSETS.items():
        sub = [r for r in test if keep(r)]
        print(f"  {name:15} n {len(sub):4}  " + "  ".join(f"{v} {auc(sub, preds[v]):.3f}" for v in args.versions))
    print("false positives caught at each version's 2% cut, by why follow-up rejected them (test):")
    for reason in sorted({r["label"].get("reason") or "?" for r in test if r["label"]["target"] == "reject"}):
        fps = [r for r in test if r["label"]["target"] == "reject" and (r["label"].get("reason") or "?") == reason]
        print(f"  {reason:15} n {len(fps):4}  " + "  ".join(
            f"{v} {np.mean([preds[v][r['toi']] < cuts[v] for r in fps]):.0%}" for v in args.versions))

    if len(args.versions) >= 2:
        a, b = args.versions[0], args.versions[-1]
        print(f"\ntest TOIs that crossed the cut from {a} to {b}:")
        for r in test:
            before, after = preds[a][r["toi"]] < cuts[a], preds[b][r["toi"]] < cuts[b]
            if before != after:
                good = after == (r["label"]["target"] == "reject")
                print(f"  TOI {r['toi']:>8} {r['label']['target']:6} ({r['label'].get('reason') or '-':11}) "
                      f"{'now rejected' if after else 'now kept':12} {'better' if good else 'worse':6} "
                      f"P {preds[a][r['toi']]:.3f} -> {preds[b][r['toi']]:.3f}")


if __name__ == "__main__":
    main()
