"""Freeze the held-back stars: 1,000 fetched survey stars that the bulk search never touches.

They are drawn uniformly at random across the survey ranking, so a candidate rate measured on the
searched stars carries over to them, and the draw happens before any search runs. The list is
committed, which timestamps it. TIC 458191673 is left out of the draw: it is the rediscovery check
and has to be searched.

Run: python -m pipeline.holdout [--manifest data/fetch_manifest.csv]
Writes survey/holdout.txt and refuses to overwrite it.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from .paths import DATA, ROOT

N_HOLDOUT = 1000
SEED = 20261027
REDISCOVERY_TIC = 458191673
OUT = ROOT / "survey" / "holdout.txt"


def draw(tics: list[int], n: int = N_HOLDOUT, seed: int = SEED) -> list[int]:
    pool = [t for t in tics if t != REDISCOVERY_TIC]
    picked = set(np.random.default_rng(seed).choice(pool, size=n, replace=False).tolist())
    return [t for t in tics if t in picked]  # kept in survey order


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", default=str(DATA / "fetch_manifest.csv"),
                    help="the fetch job's manifest.csv; stars without data are not drawn")
    args = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; the held-back set is frozen")

    with open(args.manifest, newline="", encoding="utf-8") as f:
        with_data = [int(r["tic"]) for r in csv.DictReader(f) if int(r["n_sectors"]) > 0]
    order = [int(t) for t in (DATA / "survey_order.txt").read_text().split()]
    have = set(with_data)
    held = draw([t for t in order if t in have])

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(map(str, held)) + "\n", encoding="utf-8")
    ranks = [order.index(t) + 1 for t in held]
    print(f"{len(held)} of {len(have)} stars held back (seed {SEED}); "
          f"survey ranks {min(ranks)}-{max(ranks)}, median {int(np.median(ranks))}")


if __name__ == "__main__":
    main()
