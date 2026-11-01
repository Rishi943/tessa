"""LEO-Vetter (Kunimoto et al. 2025, GPL-3.0) on every labelled TOI: a candidate input for System 1.

Its flux-level metrics (SES/MES series, trapezoid and transit fits, SWEET, odd/even, individual events,
modshift) on our own detrended light curve at our refitted ephemeris, plus which of its 13 false-alarm and
4 false-positive tests each signal fails at its default thresholds. Pixel-level vetting is left out.
Kept only if it improves System 1 on train out-of-fold and val (same test as the aperture concentration).

Run: python -m finetune.leo --export DIR            jobs for every vetted TOI (leo_jobs.json)
     (Kaggle CPU, tessa-ft-fetch mounted, leo-vetter installed)
     python -m finetune.leo --job [--part 0 --parts 3]
     python -m finetune.leo --import DIR            the job's leo_TOI.json into data/finetune/leo/
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from pipeline.paths import DATA

T0 = time.time()
G_SUN = 27400.0  # surface gravity of the Sun, cm/s^2


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min  {msg}", flush=True)


def export(folder: Path) -> None:
    jobs = []
    for p in sorted((DATA / "finetune" / "vet").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        s, st = d["checks"]["signal"], d["checks"]["star"]
        jobs.append({"toi": d["toi"], "tic": d["tic"], "sig": {k: s[k] for k in ("period", "t0", "dur")},
                     "star": {k: st.get(k) for k in ("rad", "mass", "teff")}})
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "leo_jobs.json").write_text(json.dumps(jobs), encoding="utf-8")
    print(f"{len(jobs)} TOIs")


def leo_star(star: dict) -> dict:
    from leo_vetter.stellar import quadratic_ldc
    rad = star["rad"] if star["rad"] and math.isfinite(star["rad"]) else 1.0
    mass = star["mass"] if star["mass"] and math.isfinite(star["mass"]) else rad
    teff = star["teff"] if star["teff"] and math.isfinite(star["teff"]) else 5000.0
    logg = math.log10(G_SUN * mass / rad**2)
    u1, u2 = quadratic_ldc(teff, logg)
    return {"rad": rad, "mass": mass, "Teff": teff, "e_rad": 0.1 * rad, "e_mass": 0.1 * mass, "e_Teff": 150.0,
            "rho": mass / rad**3, "logg": logg, "u1": float(u1), "u2": float(u2)}


def one(job: dict, shard: str) -> dict:
    from leo_vetter import thresholds as th
    from leo_vetter.main import TCELightCurve
    from pipeline.search import detrend
    with np.load(shard) as z:
        raw = {k: z[f"{job['tic']}_{k}"] for k in ("time", "flux", "sector")}
    lc = detrend(raw["time"], raw["flux"], raw["sector"])
    t, y, w = lc["t"], lc["y"], lc["w"]
    ok = np.isfinite(t) & np.isfinite(y) & (w > 0)
    t, y, w = t[ok], y[ok], w[ok]
    # the undetrended flux at the same times, each sector normalised: LEO-Vetter's systematics tests read it
    order = np.argsort(raw["time"])
    rt, rf, rs = raw["time"][order], raw["flux"][order].astype(float), raw["sector"][order]
    for s in np.unique(rs):
        rf[rs == s] /= np.nanmedian(rf[rs == s])
    rawb = np.interp(t, rt[np.isfinite(rf)], rf[np.isfinite(rf)])
    sig = job["sig"]
    tlc = TCELightCurve(job["tic"], t, rawb, 1 + y, 1 / np.sqrt(w), sig["period"], sig["t0"], sig["dur"])
    tlc.compute_flux_metrics(leo_star(job["star"]), verbose=False)
    m = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v) else None)
         for k, v in tlc.metrics.items() if not isinstance(v, (str, list, dict, np.ndarray))}
    fails = {}
    for case, tests in (("FA", ["weak", "invalid_transits", "bad_shape", "non_unique", "chases", "dmm", "single_event",
                                "bad_fit", "sinusoidal", "unphysical_duration", "asymmetric", "chi", "data_gapped"]),
                        ("FP", ["odd_even", "vshaped", "large", "secondary"])):
        for name in tests:
            try:
                flag, _ = getattr(th, name)(tlc.metrics, th._default_thresholds)
                fails[name] = bool(flag)
            except Exception as e:
                fails[name] = None
                m[f"error_{name}"] = f"{e.__class__.__name__}"
    return {"toi": job["toi"], "metrics": m, "fails": fails,
            "fa_fail": any(fails[n] for n in fails if n not in ("odd_even", "vshaped", "large", "secondary")),
            "fp_fail": any(fails.get(n) for n in ("odd_even", "vshaped", "large", "secondary"))}


def worker(args):
    job, shard, out = args
    path = Path(out) / f"leo_{job['toi']}.json"
    try:
        r = one(job, shard)
    except Exception as e:  # one TOI's failure must not stop the others
        r = {"toi": job["toi"], "error": f"{e.__class__.__name__}: {e}"[:200]}
    path.write_text(json.dumps(r), encoding="utf-8")
    return r


def job_main(part: int, parts: int, out: Path, lc_root: Path, max_minutes: float) -> None:
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    from finetune.inputs import shards_by_tic
    jobs = json.loads(Path("leo_jobs.json").read_text(encoding="utf-8"))[part::parts]
    shards = shards_by_tic(lc_root)
    out.mkdir(parents=True, exist_ok=True)
    done = {p.stem.removeprefix("leo_") for p in out.glob("leo_*.json")}
    todo = [(j, shards[j["tic"]], str(out)) for j in jobs if j["toi"] not in done and j["tic"] in shards]
    log(f"part {part}/{parts}: {len(todo)} of {len(jobs)} TOIs ({sum(j['tic'] not in shards for j in jobs)} without a light curve)")
    n = errors = 0
    with ProcessPoolExecutor(os.cpu_count(), mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(worker, a) for a in todo]
        for f in futures:
            if (time.time() - T0) / 60 > max_minutes:
                log("time limit: stopping; a rerun continues where this left off")
                pool.shutdown(cancel_futures=True)
                break
            r = f.result()
            n += 1
            errors += "error" in r
            if n % 100 == 0:
                log(f"{n}/{len(todo)} ({errors} errors)")
    log(f"done: {n} ({errors} errors)")


def import_results(folder: Path) -> None:
    dest = DATA / "finetune" / "leo"
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in folder.rglob("leo_*.json"):
        if p.name != "leo_jobs.json":
            (dest / p.name.removeprefix("leo_")).write_bytes(p.read_bytes())
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
    ap.add_argument("--max-minutes", type=float, default=660)
    ap.add_argument("--import", dest="results", type=Path)
    a = ap.parse_args()
    if a.export:
        export(a.export)
    elif a.job:
        job_main(a.part, a.parts, a.out, a.lc, a.max_minutes)
    else:
        import_results(a.results)


if __name__ == "__main__":
    main()
