"""The code checks for each labelled TOI, made the way the survey makes them for its own dips.

For each TOI: its host's FFI light curve (fetch.py --ffi-only), the TOI's ephemeris refined on our
own detrended light curve (so depth, SNR and timing are what our search would measure), then
pipeline.vet with blind=True (the known-object match would give the answer away) and the
centroid check on signals that pass the code rules. Stars are vetted a few at a time in separate
processes; most of the time is spent waiting on Gaia and TESScut.

Run (Kaggle, with the fetch job's output mounted; --part splits the work across jobs):
  python ft_inputs.py --labels labels.json --stars stars.json --out /kaggle/working --part 0 --parts 3
Writes vet/TOI.json (label + checks) and summary_PART.csv.
"""
import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

T0 = time.time()
GRID_PERIODS = 61
GRID_OFFSETS = 41
FIELDS = ["toi", "tic", "split", "target", "status", "code_verdict", "reasons", "snr", "transits", "depth_ppm",
          "dp_sigma", "dt0_min", "dur_ratio", "seconds"]


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min  {msg}", flush=True)


def box_snr(lc: dict, period: float, t0: float, dur: float) -> float:
    """The box search's dip SNR at one ephemeris (as search.check_peak computes it)."""
    t, y, w = lc["t"], lc["y"], lc["w"]
    inside = np.abs((t - t0 + period / 2) % period - period / 2) < dur / 2
    W, r = lc["W"], w[inside].sum()
    if not 0 < r < W:
        return 0.0
    s = np.sum(w[inside] * y[inside])
    return float(-s * W / (r * (W - r)) * np.sqrt(r * (W - r) / W))


def refine(lc: dict, toi: dict) -> tuple[float, float, float]:
    """Best period, epoch and duration near the TOI's, on our light curve: the TOI's epoch moved to the
    middle of the data, then period x epoch, then duration, then a finer epoch."""
    p, dur = toi["period"], toi["dur_h"] / 24
    t = lc["t"]
    baseline = max(t.max() - t.min(), p)
    n = round((np.median(t) - toi["t0_btjd"]) / p)
    t0 = toi["t0_btjd"] + n * p
    sp = toi["period_err"] if toi["period_err"] and math.isfinite(toi["period_err"]) else 0.0
    st0 = toi["t0_err"] if toi["t0_err"] and math.isfinite(toi["t0_err"]) else 0.0
    st0 = math.hypot(st0, n * sp)
    dp = dur / 4 * p / baseline  # moves the last transit by a quarter of a duration
    half_p = max(min(3 * sp, 0.002 * p), 2 * dp)  # a huge period error means a poor ephemeris, not a wide search
    periods = np.linspace(p - half_p, p + half_p, min(GRID_PERIODS, 2 * int(half_p / dp) + 1))
    half_t = min(max(1 / 24, 3 * st0), p / 4)
    offsets = np.linspace(-half_t, half_t, min(GRID_OFFSETS, 2 * int(half_t / (dur / 8)) + 1))
    best = max(((box_snr(lc, q, t0 + o, dur), q, t0 + o) for q in periods for o in offsets))
    _, p, t0 = best
    best = max(((box_snr(lc, p, t0, d), d) for d in dur * np.linspace(0.5, 1.6, 12)))
    dur = best[1]
    best = max(((box_snr(lc, p, t0 + o, dur), t0 + o) for o in np.linspace(-dur / 4, dur / 4, 17)))
    return p, best[1], dur


def one(job: tuple[dict, dict, str]) -> dict:
    from pipeline.search import check_peak, detrend
    from pipeline.vet import finite, vet
    toi, star, shard = job
    t_start = time.time()
    row = {"toi": toi["toi"], "tic": toi["tic"], "split": toi["split"], "target": toi["target"]}
    try:
        with np.load(shard) as z:
            raw = {k: z[f"{toi['tic']}_{k}"] for k in ("time", "flux", "sector", "source")}
        lc = detrend(raw["time"], raw["flux"], raw["sector"])
        lc["W"] = lc["w"].sum()
        p, t0, dur = refine(lc, toi)
        peak = check_peak(lc, p, t0, dur, -1)
        sig = {"period": p, "t0": t0, "dur": dur, "depth_ppm": peak["depth_ppm"], "snr": peak["snr"]}
        sp = toi["period_err"] if toi["period_err"] and math.isfinite(toi["period_err"]) and toi["period_err"] > 0 else None
        row.update(snr=round(peak["snr"], 2), transits=peak["transits"], depth_ppm=round(peak["depth_ppm"]),
                   dp_sigma=round((p - toi["period"]) / sp, 2) if sp else None,
                   dt0_min=round(((t0 - toi["t0_btjd"] + p / 2) % p - p / 2) * 1440, 1),
                   dur_ratio=round(dur * 24 / toi["dur_h"], 2))
        if peak["depth_ppm"] <= 0 or peak["transits"] < 3:
            row["status"] = "no dip"  # out of reach of the survey's gate 1 anyway
            return row
        checks = vet(toi["tic"], sig, Path(shard).parent, blind=True, centroid=True, star=star, raw=raw)
        out = {"toi": toi["toi"], "tic": toi["tic"], "label": {k: toi[k] for k in toi}, "checks": finite(checks)}
        Path(os.environ["FT_OUT"], "vet", f"{toi['toi']}.json").write_text(json.dumps(out, default=float),
                                                                           encoding="utf-8")
        row.update(status="ok", code_verdict=checks["code_verdict"], reasons="; ".join(checks["reasons"])[:300])
    except Exception as e:  # one bad star must not stop the job; it shows in the summary
        row.update(status="error", reasons=f"{e.__class__.__name__}: {e}"[:300])
    finally:
        row["seconds"] = round(time.time() - t_start, 1)
    return row


def shards_by_tic(root: Path) -> dict[int, str]:
    found = {}
    for manifest in root.rglob("manifest.csv"):
        with manifest.open(newline="") as f:
            for r in csv.DictReader(f):
                if int(r["n_sectors"]):
                    found[int(r["tic"])] = str(manifest.parent / f"lc_{int(r['shard']):03d}.npz")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--stars", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lc", default=os.environ.get("KAGGLE_INPUTS", "/kaggle/input"))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-minutes", type=float, default=660)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "vet").mkdir(parents=True, exist_ok=True)
    os.environ["FT_OUT"] = str(out)
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    stars = json.loads(Path(args.stars).read_text(encoding="utf-8"))
    shards = shards_by_tic(Path(args.lc))
    hosts = sorted({r["tic"] for r in labels})[args.part::args.parts]  # a host's TOIs stay in one part
    mine = [r for r in labels if r["tic"] in set(hosts)][:args.limit]
    rows, jobs = [], []
    for r in mine:
        star = stars.get(str(r["tic"]))
        if r["tic"] not in shards:
            rows.append({"toi": r["toi"], "tic": r["tic"], "split": r["split"], "target": r["target"], "status": "no light curve"})
        elif not star or not star.get("rad") or star.get("ra") is None:
            rows.append({"toi": r["toi"], "tic": r["tic"], "split": r["split"], "target": r["target"], "status": "no star info"})
        else:
            jobs.append((r, star, shards[r["tic"]]))
    log(f"part {args.part}/{args.parts}: {len(mine)} TOIs, {len(jobs)} to vet, {len(shards)} light curves mounted")
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for n, done in enumerate(as_completed(futures), 1):
            rows.append(done.result())
            if n % 25 == 0:
                log(f"{n}/{len(jobs)} vetted")
            if (time.time() - T0) / 60 > args.max_minutes:
                log("time limit reached; cancelling the rest")
                for f in futures:
                    f.cancel()
                break
    with (out / f"summary_{args.part}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for r in rows:
        key = r.get("code_verdict") or r.get("status")
        counts[key] = counts.get(key, 0) + 1
    log(f"done: {counts}")


if __name__ == "__main__":
    main()
