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

Run: python -m pipeline.fpp export TIC [TIC ...] --out DIR      (writes fpp_input_TIC.json)
     Kaggle: python fpp_job.py [--runs 5] [--max-minutes 600]   (reads every fpp_input_*.json beside it)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

WINDOW_DURATIONS = 3
BIN_MIN = 5
EXPTIME_S = 600
N_SIM = 1_000_000
CIRCLE_PX = 2.0  # aperture radius for stars with no TESS-SPOC sector
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


def export(tic: int, out: Path) -> Path:
    from .tools import light_curve
    cand, _ = candidate(tic)
    eph = cand["when"]["ephemeris"]
    checks, raw, lc = light_curve(tic)
    p, t0, t14 = eph["period"], eph["t0_btjd"], eph["t14_h"] / 24
    keep = np.abs(lc["span"] * 86400 - EXPTIME_S) < 60  # 10-minute points only
    dt = (lc["t"] - t0 + p / 2) % p - p / 2
    near = keep & (np.abs(dt) < WINDOW_DURATIONS * t14)
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
    d = {"tic": tic, "period": p, "t0_btjd": t0, "t14_h": eph["t14_h"],
         "depth": cand["signal"]["depth_ppm"] * 1e-6, "exptime_s": EXPTIME_S, "bin_min": BIN_MIN,
         "lc_sectors": sectors, "sources": sources,  # 0 SPOC, 1 TESS-SPOC, 2 QLP
         "geometry_sectors": sorted(s for s, src in sources.items() if src == 1),
         "n_points_unbinned": int(near.sum()), "time": t_bin, "flux": f_bin, "flux_err": float(np.median(e_bin))}
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"fpp_input_{tic}.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    print(f"TIC {tic}: {d['n_points_unbinned']} points in {len(t_bin)} bins from sectors {sectors}; "
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


def run_one(d: dict, runs: int, deadline: float, outdir: Path) -> dict:
    import triceratops.triceratops as tr
    tic = d["tic"]
    work = outdir / f"work_{tic}"
    work.mkdir(parents=True, exist_ok=True)
    aps = apertures(tic, d["geometry_sectors"], work) if d["geometry_sectors"] else {}
    sectors = sorted(aps)
    circular = not sectors
    if circular:  # QLP-only stars: a circular aperture of CIRCLE_PX pixels around the target, like QLP's own
        sectors = d["lc_sectors"]
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
    target.calc_depths(tdepth=d["depth"], all_ap_pixels=[np.array(aps[s]) for s in sectors])
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
           "neighbours_within_10px": int(len(target.stars)) - 1}
    (outdir / f"fpp_{tic}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--max-minutes", type=float, default=600)
    args = ap.parse_args()
    here = Path(__file__).resolve().parent
    outdir = Path("/kaggle/working") if Path("/kaggle/working").exists() else here / "out"
    outdir.mkdir(parents=True, exist_ok=True)
    retrying_downloads()
    deadline = T0 + args.max_minutes * 60
    inputs = sorted(here.glob("fpp_input_*.json"))
    log(f"{len(inputs)} candidates")
    for path in inputs:
        d = json.loads(path.read_text(encoding="utf-8"))
        for attempt in range(3):  # MAST and the TIC service drop connections now and then
            try:
                r = run_one(d, args.runs, deadline, outdir)
                log(f"TIC {d['tic']}: FPP {r.get('FPP_mean')}, NFPP {r.get('NFPP_mean')}, validated {r.get('validated')}")
                break
            except Exception as e:  # one candidate's failure must not stop the others
                import traceback
                traceback.print_exc()
                (outdir / f"fpp_{d['tic']}.json").write_text(json.dumps({"tic": d["tic"], "error": f"{e.__class__.__name__}: {e}"}))
                if "Connection" not in e.__class__.__name__ + str(e):
                    break
                log(f"TIC {d['tic']}: connection lost, trying again in a minute")
                time.sleep(60)


if __name__ == "__main__":
    main()
