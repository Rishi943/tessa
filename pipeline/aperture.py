"""Is the dip's light spread like the star's? A blend test on the same TESScut cutouts as the centroid.

The transit difference image (frames around each transit minus frames in it, pipeline/centroid.py) holds
the light that went missing. If the target dims, that light has the target's own image: the same share of
it falls in a 1-pixel circle as in the star's image. If a star a pixel or two away dims, the missing light
is spread wider, so less of it sits in the core: the dip looks deeper in a wide aperture than a narrow one.

  vs_psf   = (difference flux within 1 px / within 3 px) / (the same for a Gaussian fitted to the star)
  vs_image = the same against the star's image itself

Lower means the missing light is spread wider than the star's. Neither is exactly 1 on target (a Gaussian
lacks the TESS image's wings; neighbours' light widens the star's image), so the planet-like range comes
from the labelled TOIs. Errors come from resampling the transits, and sectors combine as a weighted mean.
A neighbour closer than about half a pixel looks like the target here too.

Run: python -m pipeline.aperture TIC --results results.csv       (survey stars; cutouts from the cache)
     python -m pipeline.aperture --export DIR                      (the fine-tuning TOIs, as Kaggle jobs)
     python -m pipeline.aperture --job [--part 0 --parts 3]        (on Kaggle, with aperture_jobs.json beside it)
     python -m pipeline.aperture --import DIR                      (the job's aperture_TOI.json into the TOIs' checks)
"""
import argparse
import contextlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .centroid import BOOTSTRAPS, CUTOUT_PX, MAX_SECTORS, SECTOR_JOBS, cutout, fit, gaussian, transit_images
from .paths import DATA

CORE_PX, WIDE_PX = 1.0, 3.0
RING_PX = 4.0  # the sky: pixels farther than this from the star
MIN_SNR = 3  # the difference flux in the wide circle, against its resampled error


def circles(shape, x0: float, y0: float):
    ys, xs = np.mgrid[:shape[0], :shape[1]]
    r = np.hypot(xs - x0, ys - y0)
    return r <= CORE_PX, r <= WIDE_PX


def sector_concentration(path: Path, sig: dict) -> dict | None:
    im = transit_images(path, sig)
    if im is None:
        return None
    diffs, direct = im["diffs"], im["direct"]
    centre = ((CUTOUT_PX - 1) / 2, (CUTOUT_PX - 1) / 2)
    amp, x0, y0, sigma, _ = fit(direct, centre)
    core, wide = circles(direct.shape, x0, y0)
    ys, xs = np.mgrid[:direct.shape[0], :direct.shape[1]]
    ring = np.hypot(xs - x0, ys - y0) > RING_PX
    # sky that changes between the frames in and around a transit: summed over the wide circle's 29
    # pixels it would swamp a dip of a part per thousand, so each difference image loses its ring's median
    diffs = diffs - np.nanmedian(diffs[:, ring], axis=1)[:, None, None]
    psf = gaussian([amp, x0, y0, sigma, 0.0], xs, ys)
    star = direct - np.nanmedian(direct[ring])
    expected = {"psf": psf[core].sum() / psf[wide].sum(), "image": np.nansum(star[core]) / np.nansum(star[wide])}

    def measure(d):
        return np.nansum(d[core]), np.nansum(d[wide])
    c, w = measure(diffs.mean(axis=0))
    rng = np.random.default_rng(im["sector"])
    boot = np.array([measure(diffs[rng.integers(0, len(diffs), len(diffs))].mean(axis=0)) for _ in range(BOOTSTRAPS)])
    snr = w / np.std(boot[:, 1]) if np.std(boot[:, 1]) > 0 else float("nan")
    out = {"sector": im["sector"], "transits": len(diffs), "diff_snr": float(snr) if np.isfinite(snr) else None}
    if not snr >= MIN_SNR or w <= 0:
        return {**out, "used": False}
    ratio = c / w
    pos = boot[:, 1] > 0
    err = float(np.std(boot[pos, 0] / boot[pos, 1]))
    return {**out, "used": True, "core_share": float(ratio), "err": err,
            "vs_psf": float(ratio / expected["psf"]), "vs_image": float(ratio / expected["image"]),
            "depth_core_ppm": float(1e6 * c / np.nansum(star[core])), "depth_wide_ppm": float(1e6 * w / np.nansum(star[wide]))}


def combine(per: list[dict]) -> dict:
    used = [r for r in per if r.get("used") and r["err"] > 0]
    if not used:
        return {"verdict": "inconclusive", "why": "no sector with a clear difference image", "sectors": per}
    out = {"sectors_used": len(used)}
    for key in ("vs_psf", "vs_image"):
        e = np.array([r["err"] * r[key] / r["core_share"] for r in used])  # same relative error as core_share
        v = np.array([r[key] for r in used])
        w = 1 / e**2
        mean = float(np.sum(w * v) / w.sum())
        err = float(1 / np.sqrt(w.sum()))
        if len(used) > 1:  # sectors that disagree beyond their errors widen it, as in the centroid
            err *= max(1.0, float(np.sqrt(np.sum(w * (v - mean) ** 2) / (len(used) - 1))))
        out[key] = mean
        out[f"{key}_err"] = err
    out["depth_core_ppm"] = float(np.median([r["depth_core_ppm"] for r in used]))
    out["depth_wide_ppm"] = float(np.median([r["depth_wide_ppm"] for r in used]))
    out["verdict"] = "measured"
    out["sectors"] = per
    return out


def concentration(star: dict, sig: dict, sectors: list[int], keep: bool = True) -> dict:
    """keep=False deletes each cutout after use (Kaggle: thousands of them)."""
    sectors = sorted(sectors)[-MAX_SECTORS:]

    def one(s):
        try:
            path = cutout(star["ra"], star["dec"], s)
        except Exception as e:
            return {"sector": s, "used": False, "error": f"{e.__class__.__name__}: {e}"[:120]}
        try:
            return sector_concentration(path, sig)
        except Exception as e:
            return {"sector": s, "used": False, "error": f"{e.__class__.__name__}: {e}"[:120]}
        finally:
            if not keep:
                path.unlink(missing_ok=True)
    with contextlib.redirect_stdout(sys.stderr), ThreadPoolExecutor(SECTOR_JOBS) as pool:
        per = [r for r in pool.map(one, sectors) if r]
    return combine(per)


def export(folder: Path) -> None:
    """The agent-bound fine-tuning TOIs (all splits): star position, ephemeris, the centroid's sectors."""
    stars = json.loads((DATA / "finetune" / "stars.json").read_text(encoding="utf-8"))
    jobs = []
    for p in sorted((DATA / "finetune" / "vet_s1").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        c = d["checks"]
        sectors = [s["sector"] for s in (c.get("centroid") or {}).get("sectors", [])]
        star = stars.get(str(d["tic"]))
        if c["code_verdict"] != "agent" or not sectors or not star:
            continue
        s = c["signal"]
        jobs.append({"toi": d["toi"], "star": {"ra": star["ra"], "dec": star["dec"]}, "sectors": sectors,
                     "sig": {"period": s["period"], "t0": s["t0"], "dur": s["dur"], "depth_ppm": s["depth_ppm"]}})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "aperture_jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    print(f"{len(jobs)} TOIs, {sum(len(j['sectors']) for j in jobs)} sector cutouts")


def job(part: int, parts: int, out: Path, max_minutes: float, toi_jobs: int) -> None:
    t0 = time.time()
    jobs = json.loads(Path("aperture_jobs.json").read_text(encoding="utf-8"))[part::parts]
    out.mkdir(parents=True, exist_ok=True)
    done = {p.stem.removeprefix("aperture_") for p in out.glob("aperture_*.json")}
    todo = [j for j in jobs if j["toi"] not in done]
    print(f"{time.strftime('%H:%M:%S')} part {part}/{parts}: {len(todo)} of {len(jobs)} TOIs to measure", flush=True)

    def one(j):
        if (time.time() - t0) / 60 > max_minutes:
            return None
        try:
            r = concentration(j["star"], j["sig"], j["sectors"], keep=False)
        except Exception as e:  # one TOI's failure must not stop the others
            r = {"verdict": "inconclusive", "why": f"{e.__class__.__name__}: {e}"[:160]}
        (out / f"aperture_{j['toi']}.json").write_text(json.dumps({"toi": j["toi"], **r}, default=float), encoding="utf-8")
        return r
    n = 0
    with ThreadPoolExecutor(toi_jobs) as pool:
        for r in pool.map(one, todo):
            n += r is not None
            if n and n % 50 == 0:
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(todo)} in {(time.time() - t0) / 60:.0f} min", flush=True)
    print(f"{time.strftime('%H:%M:%S')} done: {n} measured in {(time.time() - t0) / 60:.0f} min "
          f"({len(todo) - n} left for a rerun)", flush=True)


def import_results(folder: Path) -> None:
    n = 0
    for p in sorted(folder.rglob("aperture_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        vet = DATA / "finetune" / "vet_s1" / f"{r['toi']}.json"
        if vet.exists():
            d = json.loads(vet.read_text(encoding="utf-8"))
            d["checks"]["aperture"] = {k: v for k, v in r.items() if k != "toi"}
            vet.write_text(json.dumps(d), encoding="utf-8")
            n += 1
    print(f"{n} TOIs updated")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tic", type=int, nargs="?")
    ap.add_argument("--results", help="search results.csv with the signal (survey stars)")
    ap.add_argument("--export", type=Path)
    ap.add_argument("--job", action="store_true")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--toi-jobs", type=int, default=4, help="TOIs measured at once (each fetches 3 cutouts at once)")
    ap.add_argument("--import", dest="results_dir", type=Path)
    a = ap.parse_args()
    if a.export:
        export(a.export)
    elif a.job:
        job(a.part, a.parts, a.out, a.max_minutes, a.toi_jobs)
    elif a.results_dir:
        import_results(a.results_dir)
    else:
        from .centroid import survey_sectors
        from .vet import signal_from_results, star_info
        star = star_info(a.tic)
        print(json.dumps(concentration(star, signal_from_results(a.tic, Path(a.results)), survey_sectors(a.tic, star)), indent=1, default=float))


if __name__ == "__main__":
    main()
