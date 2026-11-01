"""False-positive probability for a candidate with TRICERATOPS (Giacalone et al. 2021), in two halves.

TRICERATOPS weighs "a planet transits the target" against eclipsing binaries and planets on the target,
on unresolved companions, in the background and on every TIC neighbour within 10 pixels, from the transit
shape and how much of each star's light falls in the photometric aperture. FPP is the probability of any
false positive, NFPP that the signal is on a resolved neighbour. The usual validation bar is FPP < 0.015
and NFPP < 0.001; FPP < 0.5 with NFPP < 0.001 is "likely a planet" (classify).

export (local): the folded light curve within 3 durations of mid-transit, 600-s points only (TESS-SPOC
  10-min sectors and QLP 200-s sectors binned to 10 min by search.detrend; 30-min sectors left out, so one
  exposure time fits the model), averaged in 5-minute phase bins: TRICERATOPS draws 10^6 light curves
  per scenario at every point, and a few thousand unbinned points exhaust a Kaggle machine's memory.
run (Kaggle CPU, entry fpp_job.py): apertures from the TESS-SPOC light-curve files (the only FFI product
  with one; QLP sectors give the light curve, TESS-SPOC sectors the geometry), field stars from Gaia, FFI
  cutouts through lightkurve with retries (MAST answers bursts with 503s), then N_RUNS independent runs.
  Both the TESS-SPOC and QLP fluxes are corrected for dilution, so the default dilution_corrected=True holds.

jobs (Kaggle, fetch job mounted): the same fold, done in the job from the mounted light curves, for a list of
  signals (period, epoch, duration, depth: finetune/fpp_test.py writes it for the labelled test). A signal
  with no light curve, or none at 10-minute cadence near a transit, is written as uncovered with the reason;
  every signal is written to fpp_TIC.json as it finishes, none runs past its time limit, and the job stops
  itself at --max-minutes. Signals already finished in a mounted earlier output are skipped (resume).

Run: python -m pipeline.fpp export TIC [TIC ...] --out DIR      (writes fpp_input_TIC.json)
     Kaggle: python fpp_job.py [--runs 5] [--max-minutes 600]   (reads every fpp_input_*.json beside it)
             python fpp_job.py --jobs fpp_jobs.json [--runs 1] [--limit N] [--part I --parts N]
     Laptop (TRICERATOPS venv): python -m pipeline.fpp fetch-setups --inputs DIR --cache DIR [--workers 6] [--only TIC ...]
       downloads many stars' setups at once, one process each; tools/fpp_survivors.py --cache DIR then calculates from them
"""
import argparse
import json
import os
import pickle
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np

WINDOW_DURATIONS = 3
BIN_MIN = 5
EXPTIME_S = 600
N_SIM = 1_000_000
CIRCLE_PX = 2.0  # aperture radius for stars with no TESS-SPOC sector
MIN_POINTS = 30  # 10-minute points within the window, over all transits: fewer is no light curve to fit
MAX_GEOMETRY_SECTORS = 4  # each sector costs a cutout and an aperture download, and the flux ratios are averaged
SIGNAL_LIMIT_MIN = 40  # one signal's setup and runs: MAST reads can hang with no timeout of their own
T0 = time.time()


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min  {msg}", flush=True)


def classify(fpp: float | None, nfpp: float | None) -> str:
    """Giacalone et al. 2021's classes: validated (FPP < 0.015 and NFPP < 0.001), likely planet (FPP < 0.5
    and NFPP < 0.001), likely nearby false positive (NFPP > 0.1); anything else is unclassified."""
    if fpp is None or nfpp is None:
        return "unclassified"
    if nfpp > 0.1:
        return "likely nearby false positive"
    if nfpp < 0.001:
        return "validated" if fpp < 0.015 else "likely planet" if fpp < 0.5 else "unclassified"
    return "unclassified"


# --- export (local) ----------------------------------------------------------------------------

def candidate(tic: int) -> tuple[dict, dict]:
    from .tools import BATCHES
    for vet_dir, _ in BATCHES.values():
        path = vet_dir / "candidates" / f"{tic}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8")), json.loads((vet_dir / f"{tic}.json").read_text(encoding="utf-8"))
    raise LookupError(f"TIC {tic} is not a candidate")


class Uncovered(Exception):
    """A signal that can't be given an FPP because of its data, not because something broke."""


def fold(tic: int, p: float, t0: float, t14_h: float, depth_ppm: float, raw: dict, lc: dict) -> dict:
    """The job input for one signal: its light curve folded and binned (see the module docstring).
    raw: time, flux, sector, source as the fetch jobs store them; lc: search.detrend of it."""
    t14 = t14_h / 24
    keep = np.abs(lc["span"] * 86400 - EXPTIME_S) < 60  # 10-minute points only
    dt = (lc["t"] - t0 + p / 2) % p - p / 2
    near = keep & (np.abs(dt) < WINDOW_DURATIONS * t14)
    if near.sum() < MIN_POINTS or (near & (np.abs(dt) < t14 / 2)).sum() < 3:
        raise Uncovered(f"no 10-minute light curve at the transits: {int(near.sum())} points within {WINDOW_DURATIONS} "
                        f"durations ({int(keep.sum())} 10-minute points in all, sectors {sorted({int(s) for s in lc['sector']})})")
    edges = np.arange(-WINDOW_DURATIONS * t14, WINDOW_DURATIONS * t14 + 1e-9, BIN_MIN / 1440)
    idx = np.digitize(dt[near], edges)
    w, y = lc["w"][near], lc["y"][near]
    t_bin, f_bin, e_bin = [], [], []
    for i in np.unique(idx):
        m = idx == i
        sw = w[m].sum()
        t_bin.append(float(np.sum(w[m] * dt[near][m]) / sw))
        f_bin.append(float(1 + np.sum(w[m] * y[m]) / sw))
        e_bin.append(float(1 / np.sqrt(sw)))
    sectors = sorted({int(s) for s in lc["sector"][keep]})
    sources = {int(s): int(raw["source"][raw["sector"] == s][0]) for s in np.unique(raw["sector"])}
    return {"tic": tic, "period": p, "t0_btjd": t0, "t14_h": t14_h,
            "depth": depth_ppm * 1e-6, "exptime_s": EXPTIME_S, "bin_min": BIN_MIN,
            "lc_sectors": sectors, "sources": sources,  # 0 SPOC, 1 TESS-SPOC, 2 QLP
            "geometry_sectors": sorted(s for s, src in sources.items() if src == 1),
            "n_points_unbinned": int(near.sum()), "time": t_bin, "flux": f_bin, "flux_err": float(np.median(e_bin))}


def export(tic: int, out: Path) -> Path:
    from .tools import light_curve
    cand, _ = candidate(tic)
    eph = cand["when"]["ephemeris"]
    checks, raw, lc = light_curve(tic)
    d = fold(tic, eph["period"], eph["t0_btjd"], eph["t14_h"], cand["signal"]["depth_ppm"], raw, lc)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fpp_input_{tic}.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    print(f"TIC {tic}: {d['n_points_unbinned']} points in {len(d['time'])} bins from sectors {d['lc_sectors']}; "
          f"geometry from TESS-SPOC sectors {d['geometry_sectors']}; flux error {d['flux_err'] * 1e6:.0f} ppm")
    return path


# --- run (Kaggle) ------------------------------------------------------------------------------

SPOC_FRAMES: dict[int, object] = {}  # sector -> the TESS-SPOC aperture header (celestial WCS + CCD origin)


def pixel_origin(collection) -> None:
    """TRICERATOPS 1.1 reads a cutout's first CCD column and row from the FLUX column's physical-WCS keys,
    1CRV4P and 2CRV4P, and adds them to each star's 0-based cutout pixel. TESScut cutouts since astrocut 1.x
    carry no CCD origin at all, so it is rebuilt from the sector's TESS-SPOC aperture (whose CRVAL1P/2P and
    WCS put any sky position on the CCD): origin = SPOC CCD pixel - cutout pixel, at the cutout's own
    reference point. Checked on TIC 50099421 sector 34: 1389.000, 1385.010. Sectors with no TESS-SPOC file
    (circular apertures) get origin 0: their apertures are drawn around the target in the same frame."""
    from astropy.wcs import WCS
    for tpf in collection:
        h1, h2 = tpf.hdu[1].header, tpf.hdu[2].header
        if "1CRV4P" in h1 and "2CRV4P" in h1:
            continue
        spoc = SPOC_FRAMES.get(int(tpf.hdu[0].header["SECTOR"]))
        if spoc is None:
            h1["1CRV4P"], h1["2CRV4P"] = 0, 0
            continue
        ra, dec = h2["CRVAL1"], h2["CRVAL2"]
        xs, ys = WCS(spoc).all_world2pix(ra, dec, 0)
        xc, yc = WCS(h2).all_world2pix(ra, dec, 0)
        col0, row0 = spoc["CRVAL1P"] + float(xs) - float(xc), spoc["CRVAL2P"] + float(ys) - float(yc)
        if max(abs(col0 - round(col0)), abs(row0 - round(row0))) > 0.1:
            log(f"warning: cutout origin {col0:.3f}, {row0:.3f} is not a whole pixel")
        h1["1CRV4P"], h1["2CRV4P"] = round(col0), round(row0)


def retrying_downloads():
    """lightkurve's FFI cutout downloads, retried (one lost sector stops TRICERATOPS), with the pixel
    origin keys TRICERATOPS expects."""
    import lightkurve.search as lks
    original = lks.SearchResult.download_all

    def download_all(self, *a, **k):
        for attempt in range(6):
            try:
                got = original(self, *a, **k)
                if got is not None and len(got):
                    pixel_origin(got)
                    log(f"cutout origin: column {got[0].hdu[1].header['1CRV4P']}, row {got[0].hdu[1].header['2CRV4P']}")
                    return got
            except Exception as e:
                log(f"cutout download failed ({e.__class__.__name__}), retry {attempt + 1}")
            time.sleep(20 * (attempt + 1))
        return original(self, *a, **k)

    lks.SearchResult.download_all = download_all


def apertures(tic: int, sectors: list[int], folder: Path) -> dict[int, list[list[int]]]:
    """Optimal-aperture pixels [column, row] from the TESS-SPOC light-curve files (bit 2 of APERTURE)."""
    from astropy.io import fits
    from astroquery.mast import Observations
    SPOC_FRAMES.clear()  # one candidate at a time: another star's frame for the same sector would be wrong
    obs = Observations.query_criteria(target_name=str(tic), provenance_name="TESS-SPOC", sequence_number=sectors)
    prods = Observations.filter_products(Observations.get_product_list(obs), extension="_lc.fits")
    out = {}
    for row in prods:
        path = None
        for attempt in range(4):
            try:
                res = Observations.download_products(prods[prods["dataURI"] == row["dataURI"]], download_dir=str(folder))
                path = res["Local Path"][0]
                break
            except Exception as e:
                log(f"aperture download failed ({e.__class__.__name__}), retry {attempt + 1}")
                time.sleep(15 * (attempt + 1))
        if path is None:
            continue
        with fits.open(path) as h:
            sector = int(h[0].header["SECTOR"])
            ap = h["APERTURE"]
            SPOC_FRAMES[sector] = ap.header.copy()
            rows, cols = np.where((ap.data & 2) > 0)
            out[sector] = [[int(ap.header["CRVAL1P"] + c), int(ap.header["CRVAL2P"] + r)] for r, c in zip(rows, cols)]
    return out


def thin(sectors: list[int], n: int = MAX_GEOMETRY_SECTORS) -> list[int]:
    """At most n sectors, evenly spread through the list."""
    if len(sectors) <= n:
        return sectors
    return [sectors[i] for i in np.linspace(0, len(sectors) - 1, n).round().astype(int)]


def setup_star(d: dict, work: Path):
    """The star's TRICERATOPS target with its apertures: from tri_fast's cache when the star is there, else built (TESS-SPOC
    apertures, TIC and Gaia queries, cutouts) and stored. Returns target, apertures, sectors, circular."""
    import triceratops.triceratops as tr
    tic = d["tic"]
    from . import tri_fast
    stored = tri_fast.CACHE / f"setup_{tic}.pkl" if tri_fast.CACHE else None
    if stored and stored.exists():  # the catalogues, cutouts and apertures of an earlier run on this star
        with stored.open("rb") as f:
            aps, sectors, circular, state = pickle.load(f)
        target = object.__new__(getattr(tr.target, "__wrapped__", tr.target))
        target.__dict__.update(state)
        log(f"TIC {tic}: setup from {stored}")
    else:
        aps = apertures(tic, thin(d["geometry_sectors"]), work) if d["geometry_sectors"] else {}
        sectors = sorted(aps)
        circular = not sectors
        if circular:  # QLP-only stars: a circular aperture of CIRCLE_PX pixels around the target, like QLP's own
            sectors = thin(d["lc_sectors"])
        log(f"TIC {tic}: {'circular apertures' if circular else 'TESS-SPOC apertures'} for sectors {sectors}")
        target = tr.target(ID=tic, sectors=np.array(sectors), search_radius=10)
        if circular:
            for j, s in enumerate(sectors):
                x, y = target.pix_coords[j][0]  # the target's CCD column and row in this sector
                cols, rows = np.meshgrid(np.arange(round(x) - 3, round(x) + 4), np.arange(round(y) - 3, round(y) + 4))
                m = (cols - x) ** 2 + (rows - y) ** 2 <= CIRCLE_PX ** 2
                aps[s] = np.column_stack([cols[m], rows[m]]).tolist()
        if len(target.pix_coords) != len(sectors):
            raise RuntimeError(f"TRICERATOPS placed the stars in {len(target.pix_coords)} of {len(sectors)} sectors")
        for j, s in enumerate(sectors):  # the target must sit on its own aperture, or the frames disagree
            miss = np.min(np.hypot(*(np.array(aps[s]) - target.pix_coords[j][0]).T))
            log(f"TIC {tic} sector {s}: target at CCD {target.pix_coords[j][0].round(2).tolist()}, "
                f"{miss:.2f} px from the nearest aperture pixel")
            if miss > 1.0:
                raise RuntimeError(f"sector {s}: the target is {miss:.1f} px from its aperture")
    tri_fast.settle(target)
    if stored and not stored.exists():
        with stored.with_suffix(".tmp").open("wb") as f:
            pickle.dump((aps, sectors, circular, target.__dict__), f)
        stored.with_suffix(".tmp").replace(stored)
    return target, aps, sectors, circular


def run_one(d: dict, runs: int, deadline: float, outdir: Path, scratch: Path | None = None) -> dict:
    """scratch: where the aperture files go (default beside the output; the labelled test keeps them out of it)."""
    started = time.time()
    tic = d["tic"]
    work = (scratch or outdir) / f"work_{tic}"
    work.mkdir(parents=True, exist_ok=True)
    target, aps, sectors, circular = setup_star(d, work)
    target.calc_depths(tdepth=d["depth"], all_ap_pixels=[np.array(aps[s]) for s in sectors])
    t_setup = time.time()
    t, f = np.array(d["time"]), np.array(d["flux"])
    results = []
    for i in range(runs):
        if time.time() > deadline:
            log("time limit: stopping runs")
            break
        target.calc_probs(time=t, flux_0=f, flux_err_0=d["flux_err"], P_orb=d["period"], N=N_SIM,
                          exptime=d["exptime_s"] / 86400, parallel=True, verbose=0)
        probs = target.probs
        results.append({"FPP": float(target.FPP), "NFPP": float(target.NFPP),
                        "scenarios": probs.groupby("scenario")["prob"].sum().round(6).to_dict()})
        log(f"TIC {tic} run {i + 1}/{runs}: FPP {target.FPP:.4f}, NFPP {target.NFPP:.2e}")
    fpp, nfpp = np.array([r["FPP"] for r in results]), np.array([r["NFPP"] for r in results])
    out = {"tic": tic, "triceratops": "1.1.0", "n_runs": len(results), "n_sim": N_SIM,
           "geometry_sectors": sectors, "aperture": "circular" if circular else "TESS-SPOC", "lc_sectors": d["lc_sectors"], "n_bins": len(t),
           "FPP_mean": float(fpp.mean()) if len(fpp) else None, "FPP_std": float(fpp.std()) if len(fpp) else None,
           "NFPP_mean": float(nfpp.mean()) if len(nfpp) else None, "NFPP_std": float(nfpp.std()) if len(nfpp) else None,
           "scenarios_mean": {k: float(np.mean([r["scenarios"].get(k, 0) for r in results])) for k in (results[0]["scenarios"] if results else {})},
           "class": classify(*((fpp.mean(), nfpp.mean()) if len(fpp) else (None, None))),
           "validated": bool(len(fpp) and fpp.mean() < 0.015 and nfpp.mean() < 0.001),
           "likely_planet": bool(len(fpp) and fpp.mean() < 0.5 and nfpp.mean() < 0.001), "runs": results,
           "neighbours_within_10px": int(len(target.stars)) - 1, "seconds": round(time.time() - started),
           "seconds_setup": round(t_setup - started)}
    (outdir / f"fpp_{tic}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def fetch_one(d: dict, cache: str, scratch: str) -> dict:
    """One worker process, one star: build its setup and store it in the cache (setup_TIC.pkl, gaia_TIC.csv). A failure
    comes back as a row; a dropped connection gets three tries."""
    import shutil as sh
    from . import tri_fast
    tic, started = d["tic"], time.time()
    if not tri_fast.CACHE:  # the first star this worker takes
        tri_fast.install(Path(cache))
        retrying_downloads()
    if (tri_fast.CACHE / f"setup_{tic}.pkl").exists():
        return {"tic": tic, "ok": True, "seconds": 0, "note": "already cached"}
    work = Path(scratch) / f"work_{tic}"
    error = None
    for _ in range(3):
        try:
            work.mkdir(parents=True, exist_ok=True)
            setup_star(d, work)
            error = None
            break
        except Exception as e:
            error = f"{e.__class__.__name__}: {e}"[:300]
            log(f"TIC {tic}: {error}")
            if not any(w in e.__class__.__name__ + str(e) for w in ("Connection", "RemoteService", "Timeout", "503")):
                break
            log(f"TIC {tic}: MAST or TIC service refused or dropped the request, trying again in a minute")
            time.sleep(60)
    sh.rmtree(work, ignore_errors=True)
    return {"tic": tic, "ok": error is None, "seconds": round(time.time() - started), "error": error}


def fetch_setups(inputs: Path, cache: Path, workers: int, only: list[int] | None = None) -> list[dict]:
    """The first-run setup of every fpp_input_TIC.json in inputs, downloaded by `workers` separate processes (not threads:
    SPOC_FRAMES and the lightkurve cache are per-process state) into cache, where tri_fast's runs find it. A star that fails
    is reported in the returned rows, not raised. Run with the TRICERATOPS interpreter."""
    import concurrent.futures as cf
    import multiprocessing as mp
    cache.mkdir(parents=True, exist_ok=True)
    scratch = cache / "scratch"
    jobs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(inputs.glob("fpp_input_*.json"))]
    jobs = [d for d in jobs if (not only or d["tic"] in only) and not (cache / f"setup_{d['tic']}.pkl").exists()]
    log(f"{len(jobs)} stars to fetch with {workers} workers into {cache}")
    rows = []
    with cf.ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(fetch_one, d, str(cache), str(scratch)): d["tic"] for d in jobs}
        for f in cf.as_completed(futures):
            try:
                row = f.result()
            except Exception as e:
                row = {"tic": futures[f], "ok": False, "error": f"{e.__class__.__name__}: {e}"[:300]}
            rows.append(row)
            log(f"{len(rows)}/{len(jobs)} {json.dumps(row)}")
    shutil.rmtree(scratch, ignore_errors=True)
    failed = [r["tic"] for r in rows if not r["ok"]]
    log(f"{len(rows) - len(failed)} fetched, {len(failed)} failed{': ' + str(failed) if failed else ''}")
    return rows


class TimedOut(BaseException):
    """Raised by the alarm; not an Exception, so the download retries inside the libraries can't swallow it."""


def attempt(d: dict, args, deadline: float, outdir: Path, scratch: Path | None = None) -> None:
    """One signal, written to fpp_TIC.json whatever happens: a connection error gets three tries, any other failure
    none, and nothing runs past SIGNAL_LIMIT_MIN or the job's deadline."""
    tic = d["tic"]
    for _ in range(3):  # MAST and the TIC service drop connections now and then
        left = min(SIGNAL_LIMIT_MIN * 60, deadline - time.time())
        if left <= 0:
            return log(f"TIC {tic}: no time left")
        try:
            if hasattr(signal, "SIGALRM"):
                def stop(*_):
                    raise TimedOut(f"stopped after {left / 60:.0f} min")
                signal.signal(signal.SIGALRM, stop)
                signal.alarm(int(left))
            r = run_one(d, args.runs, deadline, outdir, scratch)
            return log(f"TIC {tic}: FPP {r.get('FPP_mean')}, NFPP {r.get('NFPP_mean')}, validated {r.get('validated')}")
        except (Exception, TimedOut) as e:  # one candidate's failure must not stop the others
            import traceback
            traceback.print_exc()
            (outdir / f"fpp_{tic}.json").write_text(json.dumps({"tic": tic, "error": f"{e.__class__.__name__}: {e}"[:300]}))
            if "Connection" not in e.__class__.__name__ + str(e):
                return
            log(f"TIC {tic}: connection lost, trying again in a minute")
            time.sleep(60)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
            if scratch:  # the aperture files and the TESScut cutouts (lightkurve's cache) pile up over 100 stars
                import lightkurve
                shutil.rmtree(scratch / f"work_{tic}", ignore_errors=True)
                shutil.rmtree(lightkurve.config.get_cache_dir(), ignore_errors=True)


def folds(jobs: list[dict], inputs: Path, outdir: Path) -> dict[int, dict]:
    """Each job's folded light curve from the mounted fetch output, and coverage.json: which signals have one.
    A signal with none is written to fpp_TIC.json as uncovered, with the reason."""
    from finetune.inputs import shards_by_tic
    from .search import detrend
    shards = shards_by_tic(inputs)
    out, coverage = {}, []
    for j in jobs:
        tic = j["tic"]
        row = {"tic": tic}
        try:
            if tic not in shards:
                raise Uncovered("no light curve in the fetch output")
            with np.load(shards[tic]) as z:
                raw = {k: z[f"{tic}_{k}"] for k in ("time", "flux", "sector", "source")}
            d = fold(tic, j["period"], j["t0"], j["t14_h"], j["depth_ppm"], raw, detrend(raw["time"], raw["flux"], raw["sector"]))
            out[tic] = d
            row |= {"covered": True, "lc_sectors": d["lc_sectors"], "n_points": d["n_points_unbinned"], "n_bins": len(d["time"]),
                    "spoc_sectors": d["geometry_sectors"]}
        except Uncovered as e:
            row |= {"covered": False, "why": str(e)}
            (outdir / f"fpp_{tic}.json").write_text(json.dumps({"tic": tic, "uncovered": str(e)}), encoding="utf-8")
        except Exception as e:
            row |= {"covered": False, "why": f"{e.__class__.__name__}: {e}"[:300]}
        coverage.append(row)
    (outdir / "coverage.json").write_text(json.dumps(coverage), encoding="utf-8")
    log(f"{len(out)} of {len(jobs)} signals have a 10-minute light curve at their transits; "
        f"{sum(bool(r.get('spoc_sectors')) for r in coverage)} also have TESS-SPOC apertures")
    return out


def finished(inputs: Path) -> set[int]:
    """Signals with an FPP in a mounted earlier output (a rehearsal, or a part that hit its time cap)."""
    done = set()
    for p in inputs.rglob("fpp_[0-9]*.json"):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("FPP_mean") is not None:
                done.add(int(p.stem.removeprefix("fpp_")))
        except ValueError:
            pass
    return done


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("tics", nargs="+", type=int)
        ap.add_argument("--out", type=Path, required=True)
        args = ap.parse_args()
        for tic in args.tics:
            export(tic, args.out)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "fetch-setups":
        ap = argparse.ArgumentParser(prog="fpp fetch-setups", description=fetch_setups.__doc__)
        ap.add_argument("cmd")
        ap.add_argument("--inputs", type=Path, required=True, help="folder with fpp_input_TIC.json files")
        ap.add_argument("--cache", type=Path, required=True, help="tri_fast's setup cache (--cache of tools/fpp_survivors.py)")
        ap.add_argument("--workers", type=int, default=6)
        ap.add_argument("--only", type=int, nargs="+")
        args = ap.parse_args()
        rows = fetch_setups(args.inputs, args.cache, args.workers, args.only)
        (args.cache / "fetch_setups.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        return
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-minutes", type=float, default=600)
    ap.add_argument("--jobs", type=Path, help="fpp_jobs.json: fold these signals' light curves here instead of reading "
                                              "fpp_input_*.json")
    ap.add_argument("--limit", type=int, help="run at most this many signals (a rehearsal)")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--inputs", type=Path, default=Path(os.environ.get("KAGGLE_INPUTS", "/kaggle/input")))
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    kaggle = Path("/kaggle/working").exists()
    outdir = Path("/kaggle/working") if kaggle else here / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    retrying_downloads()
    deadline = T0 + args.max_minutes * 60
    if args.jobs:
        scratch = Path("/kaggle/temp/fpp") if kaggle else outdir / "scratch"  # aperture files stay out of the output
        scratch.mkdir(parents=True, exist_ok=True)
        jobs = json.loads(args.jobs.read_text(encoding="utf-8"))[args.part::args.parts]
        found, done = folds(jobs, args.inputs, outdir), finished(args.inputs)
        todo = [found[j["tic"]] for j in jobs if j["tic"] in found and j["tic"] not in done][:args.limit]
        log(f"{len(todo)} to run ({len(done)} finished in a mounted earlier output), {args.runs} run(s) each")
        for n, d in enumerate(todo, 1):
            attempt(d, args, deadline, outdir, scratch)
            log(f"{n}/{len(todo)} done")
        return
    inputs = sorted(here.glob("fpp_input_*.json"))
    log(f"{len(inputs)} candidates")
    for path in inputs:
        attempt(json.loads(path.read_text(encoding="utf-8")), args, deadline, outdir)


if __name__ == "__main__":
    main()
