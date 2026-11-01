"""Build demo/data, the one-star bundle tools/demo.py runs on, from the survey's data (run on the machine that has data/).

  python tools/demo_pack.py                        TIC 71503196 (a likely planet) and 240488148 (fails the season test)
  python tools/demo_pack.py TIC [TIC ...] [--out demo/data]

For each star: its row of targets.parquet, its light curves cut out of their shard (as fetched from MAST), its Gaia
neighbours and TESScut cutouts, the catalogue lists, System 1's model, the gate 1 bar of its batch, Glimmer's saved
dossier (for --replay) and its TRICERATOPS result. Then a trial run of tools/demo.py --replay on a copy, in its own
process, so anything the pipeline still downloads (cutouts, Gaia tables) is kept in the bundle too, and the time of
each step is known. Nothing in data/ is written.
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"  # the real data: read only
CACHE = DATA / "cache"
LC = {"survey": CACHE / "lc", "batch2": CACHE / "lc_batch2"}
CATALOG_FILES = ("toi.csv", "ctoi.csv", "confirmed.csv", "tess_ebs.csv", "planets.csv", "planets_orbits.csv",
                 "open_tois.csv", "open_tois_orbits.csv", "reported.csv")  # pipeline/revet.py's list
SYSTEM1_FILES = ("xgb-clues.json", "calibration-clues.json", "results-clues.json", "preds-clues.json",
                 "similar-clues.npz", "similar-clues.json", "params.json")
NOTES = {71503196: "likely planet; XGBoost ranked it 0.09, Glimmer passed it, TRICERATOPS backed the call",
         240488148: "a candidate in the first run; now flagged by the season test"}
DEFAULT = (71503196, 240488148)


def batch_of(tic: int) -> str:
    for batch in LC:
        if (DATA / "vet" / f"{batch}_v10" / f"{tic}.json").exists() or (DATA / "vet" / batch / f"{tic}.json").exists():
            return batch
    raise LookupError(f"TIC {tic} is in neither batch")


def manifest_row(tic: int, lc_dir: Path) -> tuple[list[str], dict]:
    for m in (lc_dir / "manifest.csv", DATA / "fetch_manifest.csv"):
        if m.exists():
            with m.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    if int(row["tic"]) == tic:
                        return r.fieldnames, row
    raise LookupError(f"TIC {tic} is in no light-curve manifest")


def fpp(tic: int) -> dict | None:
    best = None
    for res in sorted((DATA / "survey_v10").glob("fpp_out*/results.json")):
        for r in json.loads(res.read_text(encoding="utf-8")):
            if int(r["tic"]) == tic and r.get("ran"):
                best = {"FPP": r["FPP"], "NFPP": r["NFPP"], "class": r["class"]}
    return best


def copy(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        print(f"  missing, skipped: {src.relative_to(DATA)}")


def build(tics: list[int], out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    (out / "cache").mkdir(parents=True)
    meta = {"stars": {}}
    pd.read_parquet(DATA / "targets.parquet").query("tic in @tics").to_parquet(out / "targets.parquet")
    for name in CATALOG_FILES:
        copy(DATA / "catalogs" / name, out / "catalogs" / name)
    for name in SYSTEM1_FILES:
        copy(DATA / "finetune" / "system1" / name, out / "finetune" / "system1" / name)
    copy(DATA / "finetune" / "labels.json", out / "finetune" / "labels.json")
    stars = pd.read_parquet(out / "targets.parquet")
    lcs: dict[str, dict] = {}
    rows: dict[str, list] = {}
    for tic in tics:
        batch = batch_of(tic)
        lc_dir = LC[batch]
        fields, row = manifest_row(tic, lc_dir)
        with np.load(lc_dir / f"lc_{int(row['shard']):03d}.npz") as z:
            for k in ("time", "flux", "sector", "source"):
                lcs.setdefault(batch, {})[f"{tic}_{k}"] = z[f"{tic}_{k}"]
        rows.setdefault(batch, [fields])
        rows[batch].append({**row, "shard": "0"})
        copy(DATA / f"{batch}_gate1.json", out / f"{batch}_gate1.json")
        copy(CACHE / "gaia_neighbours" / f"{tic}.csv", out / "cache" / "gaia_neighbours" / f"{tic}.csv")
        s = stars.query("tic == @tic").iloc[0]
        for p in (CACHE / "tesscut").glob(f"s*_{s['ra']:.6f}_{s['dec']:.6f}_*.npz"):
            copy(p, out / "cache" / "tesscut" / p.name)
        copy(DATA / "dossiers" / f"{tic}.glimmer-vetter-v10.v10.json", out / "dossiers" / f"{tic}.glimmer-vetter-v10.v10.json")
        meta["stars"][str(tic)] = {"batch": batch, "note": NOTES.get(tic, ""), "triceratops": fpp(tic)}
    for batch, arrays in lcs.items():
        d = out / "cache" / LC[batch].name
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "lc_000.npz", **arrays)
        fields, *rs = rows[batch]
        with (d / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rs)
    # centroid.survey_sectors reads data/fetch_manifest.csv first, whatever the batch: the demo stars' rows
    fields = next(iter(rows.values()))[0]
    with (out / "fetch_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore")
        w.writeheader()
        for _, *rs in rows.values():
            w.writerows(rs)
    (out / "demo.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def trial(tics: list[int], out: Path) -> None:
    """tools/demo.py --replay on a copy, each star in its own process; keep whatever it downloaded into the bundle."""
    keep = ("cache", "catalogs")  # downloads worth bundling; vet/, evidence/, validation/ are the run's own output
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "data"
        shutil.copytree(out, work)
        before = {p.relative_to(work) for p in work.rglob("*") if p.is_file()}
        for tic in tics:
            t = time.time()
            print(f"\n=== trial run, TIC {tic} ===", flush=True)
            res = subprocess.run([sys.executable, str(ROOT / "tools" / "demo.py"), str(tic), "--replay", "--data", str(work)], cwd=ROOT)
            print(f"=== TIC {tic}: exit {res.returncode}, {time.time() - t:.0f} s ===", flush=True)
        new = [p for p in work.rglob("*") if p.is_file() and p.relative_to(work) not in before
               and p.relative_to(work).parts[0] in keep]
        for p in new:
            copy(p, out / p.relative_to(work))
        print(f"\nkept {len(new)} downloaded files in the bundle: " + ", ".join(str(p.relative_to(work)) for p in new[:12]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tics", type=int, nargs="*", default=list(DEFAULT))
    ap.add_argument("--out", type=Path, default=ROOT / "demo" / "data")
    ap.add_argument("--no-trial", action="store_true", help="skip the trial run")
    a = ap.parse_args()
    build(a.tics, a.out)
    if not a.no_trial:
        trial(a.tics, a.out)
    files = [p for p in a.out.rglob("*") if p.is_file()]
    print(f"\n{a.out}: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB")
    for p in sorted(files, key=lambda p: -p.stat().st_size)[:10]:
        print(f"  {p.stat().st_size / 1e6:7.2f} MB  {p.relative_to(a.out)}")


if __name__ == "__main__":
    main()
