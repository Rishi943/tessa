"""Is the dip on the target? Difference imaging on TESS full-frame-image cutouts (TESScut).

For each transit, the frames in transit are subtracted from the frames just before and after it,
which removes slow drifts and scattered light; the differences are averaged over the transits in
a sector. If the target makes the dip, that difference image is the target's own image; if a
neighbour makes it, it sits on the neighbour. Gaussian fits to the difference image and to the
out-of-transit image give the offset between them, resampling the transits gives its error, and
sectors are combined on the sky because each camera's pixels point a different way.

Run: python -m pipeline.centroid TIC --results results.csv [--sectors 6]
Prints JSON; cutouts are cached in data/cache/tesscut/.
"""
import argparse
import contextlib
import csv
import json
import sys
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from scipy.optimize import least_squares

from .paths import CACHE, DATA
from .vet import PLAUSIBLE_ECLIPSE, gaia_field, signal_from_results, star_info

CUTOUT_PX = 11
FIT_HALF = 3  # fit a 7x7 window
BOOTSTRAPS = 100
MAX_SECTORS = 6
IN_CORE = 0.4  # in-transit frames: within this fraction of the duration from mid-transit
OUT_GAP, OUT_MIN_D = 0.75, 0.125  # out-of-transit frames: from 0.75 durations out, for at least 3 h
ON_TARGET_SIGMA = 3.0
BTJD_2016 = 388.5  # 2016.0 (Gaia DR3's epoch) in BTJD


def cutout(ra: float, dec: float, sector: int) -> Path:
    folder = CACHE / "tesscut"
    hits = list(folder.glob(f"tess-s{sector:04d}-*_{ra:.6f}_{dec:.6f}_{CUTOUT_PX}x{CUTOUT_PX}_astrocut.fits"))
    if hits:
        return hits[0]
    from astroquery.mast import Tesscut
    folder.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(sys.stderr):
        got = Tesscut.download_cutouts(coordinates=SkyCoord(ra, dec, unit="deg"), size=CUTOUT_PX,
                                       sector=sector, path=str(folder))
    return Path(got["Local Path"][0])


def gaussian(params, x, y):
    amp, x0, y0, sigma, bkg = params
    return amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma**2)) + bkg


def fit(image: np.ndarray, centre: tuple[float, float], sigma: float | None = None) -> np.ndarray:
    """Gaussian fit in a window around centre (x, y); sigma fixed when given."""
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    ys, xs = np.mgrid[cy - FIT_HALF:cy + FIT_HALF + 1, cx - FIT_HALF:cx + FIT_HALF + 1]
    ok = (xs >= 0) & (ys >= 0) & (xs < image.shape[1]) & (ys < image.shape[0])
    xs, ys = xs[ok], ys[ok]
    z = image[ys, xs]
    good = np.isfinite(z)
    xs, ys, z = xs[good], ys[good], z[good]
    start = [np.nanmax(z) - np.nanmedian(z), centre[0], centre[1], sigma or 0.8, np.nanmedian(z)]
    lo = [-np.inf, cx - FIT_HALF, cy - FIT_HALF, 0.3, -np.inf]
    hi = [np.inf, cx + FIT_HALF, cy + FIT_HALF, 3.0, np.inf]
    if sigma is not None:
        def resid(p):
            return gaussian([p[0], p[1], p[2], sigma, p[3]], xs, ys) - z
        r = least_squares(resid, [start[0], start[1], start[2], start[4]],
                          bounds=([lo[0], lo[1], lo[2], lo[4]], [hi[0], hi[1], hi[2], hi[4]]))
        return np.array([r.x[0], r.x[1], r.x[2], sigma, r.x[3]])
    r = least_squares(lambda p: gaussian(p, xs, ys) - z, start, bounds=(lo, hi))
    return r.x


def sector_offset(path: Path, sig: dict) -> dict | None:
    with fits.open(path) as h:
        d = h[1].data
        t, flux, q = np.asarray(d["TIME"], float), np.asarray(d["FLUX"], float), np.asarray(d["QUALITY"])
        wcs = WCS(h[2].header)
        sector = int(h[0].header["SECTOR"])
    good = (q == 0) & np.isfinite(t) & np.isfinite(flux).reshape(len(t), -1).any(axis=1)
    t, flux = t[good], flux[good]
    p, dur = sig["period"], sig["dur"]
    epochs = np.arange(np.ceil((t.min() - sig["t0"]) / p), np.floor((t.max() - sig["t0"]) / p) + 1)
    diffs = []
    for e in epochs:
        dt = t - (sig["t0"] + e * p)
        inside = np.abs(dt) < IN_CORE * dur
        around = (np.abs(dt) > OUT_GAP * dur) & (np.abs(dt) < OUT_GAP * dur + max(dur, OUT_MIN_D))
        if inside.sum() >= 1 and (dt[around] < 0).sum() >= 1 and (dt[around] > 0).sum() >= 1:
            diffs.append(np.nanmean(flux[around], axis=0) - np.nanmean(flux[inside], axis=0))
    if len(diffs) < 3:
        return None
    diffs = np.array(diffs)
    direct = np.nanmedian(flux, axis=0)

    centre = ((CUTOUT_PX - 1) / 2, (CUTOUT_PX - 1) / 2)
    d_fit = fit(direct, centre)
    sigma = d_fit[3]
    diff_fit = fit(diffs.mean(axis=0), d_fit[1:3], sigma)
    rng = np.random.default_rng(sector)
    boot = np.array([fit(diffs[rng.integers(0, len(diffs), len(diffs))].mean(axis=0), d_fit[1:3], sigma)[:3]
                     for _ in range(BOOTSTRAPS)])
    amp_err = np.std(boot[:, 0])
    if diff_fit[0] <= 0 or diff_fit[0] < 3 * amp_err:
        return {"sector": sector, "transits": len(diffs), "diff_snr": float(diff_fit[0] / amp_err), "used": False}

    # Offsets on the sky (arcsec east, north) of the difference image from the direct image.
    def sky(x, y):
        c = wcs.pixel_to_world(x, y)
        return c.ra.deg, c.dec.deg
    ra_d, dec_d = sky(d_fit[1], d_fit[2])
    cosd = np.cos(np.radians(dec_d))
    def offset(x, y):
        ra, dec = sky(x, y)
        return np.array([(ra - ra_d) * cosd * 3600, (dec - dec_d) * 3600])
    off = offset(diff_fit[1], diff_fit[2])
    cloud = np.array([offset(b[1], b[2]) for b in boot])
    err = float(np.sqrt(np.mean(np.var(cloud, axis=0))))
    return {"sector": sector, "transits": len(diffs), "diff_snr": float(diff_fit[0] / amp_err), "used": True,
            "east_arcsec": float(off[0]), "north_arcsec": float(off[1]), "err_arcsec": err,
            "mid_btjd": float(np.median(t))}


def where(tic: int, sig: dict, max_sectors: int = MAX_SECTORS) -> dict:
    star = star_info(tic)
    with open(DATA / "fetch_manifest.csv", newline="", encoding="utf-8") as f:
        row = next((r for r in csv.DictReader(f) if int(r["tic"]) == tic), None)
    if row is None:  # stars outside the survey: whatever TESScut has
        from astroquery.mast import Tesscut
        sectors = [int(s) for s in Tesscut.get_sectors(coordinates=SkyCoord(star["ra"], star["dec"], unit="deg"))["sector"]]
    else:
        sectors = [int(s) for s in row["sectors"].split()]
    sectors = sorted(sectors)[-max_sectors:]  # the latest have the shortest cadence
    field = gaia_field(tic, star)
    if field is None:
        return {"verdict": "inconclusive", "why": "target not found in Gaia"}
    df, me = field

    per = []
    for s in sectors:
        try:
            r = sector_offset(cutout(star["ra"], star["dec"], s), sig)
        except Exception as e:
            r = {"sector": s, "used": False, "error": f"{e.__class__.__name__}: {e}"[:120]}
        if r:
            per.append(r)
    used = [r for r in per if r.get("used")]
    if not used:
        return {"verdict": "inconclusive", "why": "no sector with a clear difference image", "sectors": per}

    w = np.array([1 / r["err_arcsec"] ** 2 for r in used])
    east = float(np.sum(w * [r["east_arcsec"] for r in used]) / w.sum())
    north = float(np.sum(w * [r["north_arcsec"] for r in used]) / w.sum())
    err = float(1 / np.sqrt(w.sum()))
    mid = float(np.mean([r["mid_btjd"] for r in used]))

    # Where each plausible neighbour would put the difference image, at the data's epoch.
    years = (mid - BTJD_2016) / 365.25
    def pos(r):
        dec = r["dec"] + np.nan_to_num(r["pmdec"]) * years / 3.6e6
        ra = r["ra"] + np.nan_to_num(r["pmra"]) * years / 3.6e6 / np.cos(np.radians(r["dec"]))
        return ra, dec
    ra_me, dec_me = pos(me)
    need = sig["depth_ppm"] * 1e-6 * 10 ** (0.4 * (df["mag"] - me["mag"]))
    suspects = []
    for i, r in df[(need <= PLAUSIBLE_ECLIPSE) & (df.index != me.name)].iterrows():
        ra, dec = pos(r)
        e_n = (ra - ra_me) * np.cos(np.radians(dec_me)) * 3600
        n_n = (dec - dec_me) * 3600
        suspects.append({"source_id": int(df.at[i, "source_id"]), "sep_arcsec": round(float(np.hypot(e_n, n_n)), 1),
                         "eclipse_needed": round(float(need[i]), 4),
                         "sigma_from_centroid": round(float(np.hypot(east - e_n, north - n_n) / err), 1)})
    suspects.sort(key=lambda s: s["sigma_from_centroid"])

    off = float(np.hypot(east, north))
    sigma = off / err
    near = [s for s in suspects if s["sigma_from_centroid"] < ON_TARGET_SIGMA]
    if sigma < ON_TARGET_SIGMA and not near:
        verdict, why = "on_target", "the dip sits on the target and on none of the stars that could fake it"
    elif sigma >= ON_TARGET_SIGMA:
        verdict = "off_target"
        why = (f"the dip sits {off:.1f}\" from the target ({sigma:.1f} sigma)"
               + (f", on Gaia {near[0]['source_id']} {near[0]['sep_arcsec']}\" away" if near else ""))
    else:
        verdict = "inconclusive"
        why = f"consistent with the target but also with {len(near)} neighbour(s) that could fake it"
    return {"verdict": verdict, "why": why, "offset_arcsec": off, "err_arcsec": err, "offset_sigma": sigma,
            "east_arcsec": east, "north_arcsec": north, "sectors_used": len(used),
            "suspects": suspects[:5], "sectors": per}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tic", type=int)
    ap.add_argument("--results", required=True, help="search results.csv with this star's signal")
    ap.add_argument("--sectors", type=int, default=MAX_SECTORS, help="use at most this many (latest) sectors")
    args = ap.parse_args()
    out = where(args.tic, signal_from_results(args.tic, Path(args.results)), args.sectors)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
