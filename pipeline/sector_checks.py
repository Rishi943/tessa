"""Odd/even and secondary eclipse sector by sector: an eclipsing binary shows them in every sector, a
systematic in one. The whole-light-curve tests (pipeline/vet.py) can hide a strong difference in one
season under steady others, or show one made by a single bad sector.

For each sector with transits: vet's odd/even (depths, sigma with that sector's red noise) and secondary
(the best dimming away from the transit, and at phase 0.5). Summaries: the largest per-sector odd/even
sigma, how many sectors pass 3 sigma, the share of sectors whose odd/even difference has the same sign
as the whole curve's, and the same for a secondary at phase 0.5.

Run: python -m pipeline.sector_checks --export DIR        jobs for every vetted fine-tuning TOI
     python -m pipeline.sector_checks --job [--part 0 --parts 3]   (Kaggle CPU, tessa-ft-fetch mounted)
     python -m pipeline.sector_checks --import DIR        into data/finetune/sector_checks/
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from .paths import DATA
from .vet import odd_even, phase_offset, secondary

MIN_POINTS_IN = 3  # in-transit points a sector needs, in odd and in even transits alike
KEYS = ("sectors", "oe_max_sigma", "oe_sectors_over_3", "oe_same_sign_share", "sec_half_max_snr",
        "sec_half_sectors_over_3", "sec_half_positive_share")  # what System 1 reads (its "sector" clues)


def sub(lc: dict, mask: np.ndarray) -> dict:
    n = len(lc["t"])
    return {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in lc.items()}


def per_sector(lc: dict, sig: dict) -> dict:
    whole = odd_even(lc, sig)
    sign = np.sign((whole["depth_odd_ppm"] or 0) - (whole["depth_even_ppm"] or 0))
    inside = np.abs(phase_offset(lc["t"], sig["period"], sig["t0"])) < sig["dur"] / 2
    epoch = np.round((lc["t"] - sig["t0"]) / sig["period"]).astype(np.int64)
    rows = []
    for s in np.unique(lc["sector"]):
        m = lc["sector"] == s
        if (inside & m & (epoch % 2 == 1)).sum() < MIN_POINTS_IN or (inside & m & (epoch % 2 == 0)).sum() < MIN_POINTS_IN:
            continue
        part = sub(lc, m)
        oe, sec = odd_even(part, sig), secondary(part, sig)
        rows.append({"sector": int(s), "oe_sigma": float(oe["diff_sigma"]) if np.isfinite(oe["diff_sigma"]) else None,
                     "oe_sign": float(np.sign(oe["depth_odd_ppm"] - oe["depth_even_ppm"])),
                     "sec_snr": float(sec["snr"]), "sec_half_snr": float(sec["snr_at_half"]) if np.isfinite(sec["snr_at_half"]) else None})
    oes = [r["oe_sigma"] for r in rows if r["oe_sigma"] is not None]
    halves = [r["sec_half_snr"] for r in rows if r["sec_half_snr"] is not None]
    return {"sectors": len(rows),
            "oe_max_sigma": max(oes) if oes else None,
            "oe_sectors_over_3": sum(x >= 3 for x in oes),
            "oe_same_sign_share": float(np.mean([r["oe_sign"] == sign for r in rows])) if rows and sign else None,
            "sec_half_max_snr": max(halves) if halves else None,
            "sec_half_sectors_over_3": sum(x >= 3 for x in halves),
            "sec_half_positive_share": float(np.mean([x > 0 for x in halves])) if halves else None,
            "per_sector": rows}


def export(folder: Path) -> None:
    jobs = []
    for p in sorted((DATA / "finetune" / "vet").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d["checks"]["signal"]
        jobs.append({"toi": d["toi"], "tic": d["tic"], "sig": {k: s[k] for k in ("period", "t0", "dur")}})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "sector_jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    print(f"{len(jobs)} TOIs")


def job(part: int, parts: int, out: Path, lc_root: Path) -> None:
    from finetune.inputs import shards_by_tic
    from .search import detrend
    t0 = time.time()
    jobs = json.loads(Path("sector_jobs.json").read_text(encoding="utf-8"))[part::parts]
    shards = shards_by_tic(lc_root)
    out.mkdir(parents=True, exist_ok=True)
    n = errors = 0
    for j in jobs:
        if j["tic"] not in shards:
            continue
        try:
            with np.load(shards[j["tic"]]) as z:
                raw = {k: z[f"{j['tic']}_{k}"] for k in ("time", "flux", "sector")}
            r = per_sector(detrend(raw["time"], raw["flux"], raw["sector"]), j["sig"])
        except Exception as e:  # one TOI's failure must not stop the others
            r = {"error": f"{e.__class__.__name__}: {e}"[:200]}
            errors += 1
        (out / f"sector_{j['toi']}.json").write_text(json.dumps({"toi": j["toi"], **r}, default=float), encoding="utf-8")
        n += 1
        if n % 200 == 0:
            print(f"{time.strftime('%H:%M:%S')} {n}/{len(jobs)} ({errors} errors) in {(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"{time.strftime('%H:%M:%S')} done: {n} ({errors} errors) in {(time.time() - t0) / 60:.0f} min", flush=True)


def import_results(folder: Path) -> None:
    dest = DATA / "finetune" / "sector_checks"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in folder.rglob("sector_*.json"):
        if p.name != "sector_jobs.json":
            (dest / p.name.removeprefix("sector_")).write_bytes(p.read_bytes())
            n += 1
    print(f"{n} TOIs in {dest}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", type=Path)
    ap.add_argument("--job", action="store_true")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--lc", type=Path, default=Path(os.environ.get("KAGGLE_INPUTS", "/kaggle/input")))
    ap.add_argument("--import", dest="results", type=Path)
    a = ap.parse_args()
    if a.export:
        export(a.export)
    elif a.job:
        job(a.part, a.parts, a.out, a.lc)
    else:
        import_results(a.results)


if __name__ == "__main__":
    main()
