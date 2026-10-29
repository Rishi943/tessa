"""Score vetters on the held-out TOIs: what follow-up found against what each vetter said.

Rows: the code rules alone (vet's rejections, then gate 2's planet-like ranges: in range -> gate2, else
flag), each model's dossier verdict, and "rules + model" (a candidate needs both; either rejection
rejects). Code-rejected TOIs count for every row, as they never reach a model.
The figure that matters most: planets rejected (lost for good). Then false positives caught or leaked.

Run: python -m finetune.evaluate [--split test] [--vet vet_s1] [--models base=data/finetune/traces ...]
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline.gate2 import outside_planet_range
from pipeline.paths import DATA

FOLDER = DATA / "finetune"
VERDICTS = ("gate2", "flag", "reject")


def rules(checks: dict) -> str:
    if checks["code_verdict"] == "reject":
        return "reject"
    return "flag" if outside_planet_range(checks) else "gate2"


def combine(rule: str, model: str | None) -> str | None:
    if rule == "reject" or model == "reject":
        return "reject"
    if model is None or model == "unparseable":
        return None
    return "gate2" if rule == "gate2" and model == "gate2" else "flag"


def model_verdicts(folder: Path, sample: int) -> dict[str, str]:
    out = {}
    for path in folder.glob(f"*.{sample}.json"):
        t = json.loads(path.read_text(encoding="utf-8"))
        out[t["toi"]] = t["verdict"]
    return out


def table(rows: list[tuple[str, str]]) -> str:
    """rows: (target, predicted). Counts per outcome, and the shares that matter."""
    c = Counter(rows)
    by_target = Counter(t for t, _ in rows)
    lines = [f"{'outcome':>10} | " + " ".join(f"{v:>7}" for v in VERDICTS) + " |   n"]
    for t in VERDICTS:
        if by_target[t]:
            lines.append(f"{t:>10} | " + " ".join(f"{c[(t, v)]:>7}" for v in VERDICTS) + f" | {by_target[t]:>3}")
    planets, fps = by_target["gate2"], by_target["reject"]
    lines.append(f"planets rejected {c[('gate2', 'reject')]}/{planets} ({c[('gate2', 'reject')] / max(planets, 1):.1%}), "
                 f"planets passed {c[('gate2', 'gate2')] / max(planets, 1):.1%}; "
                 f"false positives caught {c[('reject', 'reject')] / max(fps, 1):.1%}, "
                 f"passed as planets {c[('reject', 'gate2')] / max(fps, 1):.1%}; "
                 f"exact {sum(c[(v, v)] for v in VERDICTS) / max(len(rows), 1):.1%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="*", default=[f"base={FOLDER / 'traces'}"],
                    help="NAME=FOLDER of dossier files (TOI.SAMPLE.json); sample 0 is scored")
    ap.add_argument("--subset", choices=["all", "km", "recent"], default="all")
    ap.add_argument("--vet", default="vet", help="folder in data/finetune with the vet files (vet_s1: rules with System 1)")
    args = ap.parse_args()

    tois = []
    for path in (FOLDER / args.vet).glob("*.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        lab = d["label"]
        if lab["split"] != args.split:
            continue
        if args.subset == "km" and not (lab.get("teff") or 9999) < 5300:
            continue
        if args.subset == "recent" and not lab.get("recent"):
            continue
        tois.append((d["toi"], lab, d["checks"]))
    models = {name: model_verdicts(Path(folder), 0) for name, folder in (m.split("=", 1) for m in args.models)}
    print(f"{len(tois)} {args.split} TOIs ({args.subset}); "
          f"code-rejected {sum(c['code_verdict'] == 'reject' for _, _, c in tois)}")

    results = defaultdict(list)
    missing = Counter()
    for toi, lab, checks in tois:
        rule = rules(checks)
        results["rules alone"].append((lab["target"], rule))
        for name, verdicts in models.items():
            v = "reject" if checks["code_verdict"] == "reject" else verdicts.get(toi)
            if v is None:
                missing[name] += 1
                continue
            results[name].append((lab["target"], v if v in VERDICTS else "flag"))
            both = combine(rule, v)
            if both:
                results[f"rules + {name}"].append((lab["target"], both))
    for name, rows in results.items():
        print(f"\n== {name} ({len(rows)} TOIs{f', {missing[name]} without a dossier' if missing[name] else ''})")
        print(table(rows))

    # what the rules can't see: false positives that only spectra showed
    spectra = [(lab["target"], rules(c)) for _, lab, c in tois if lab.get("reason") == "spectra"]
    if spectra:
        print(f"\nfalse positives found only by spectra: {len(spectra)}, rules passed "
              f"{sum(p == 'gate2' for _, p in spectra)} as planet-like")


if __name__ == "__main__":
    main()
