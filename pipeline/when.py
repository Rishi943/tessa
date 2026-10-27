"""When can we check? A signal's ephemeris carried forward to future TESS sectors and to nights
when telescopes on the ground can see a transit.

The ephemeris is refitted from one mid-transit time per sector (a trapezoid of the found shape
slid across that sector's data), so its errors come from the data rather than the search grid,
and they grow with every orbit predicted. Future sectors come from tess-point's pointing table;
TESScut serves the full-frame images about RELEASE_LAG_D after each orbit. Ground windows are
listed for the Las Cumbres Observatory network, whose 0.4-1 m telescopes students and amateurs
can book.

Run: python -m pipeline.when TIC --results results.csv [--lc data/cache/lc] [--days 90]
Prints JSON.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time

from .paths import CACHE
from .search import detrend
from .vet import load_lc, phase_offset, red_noise, shape, signal_from_results, star_info, trapezoid

BTJD = 2457000.0
SECTOR_HALF_D = 13.7
RELEASE_LAG_D = 10  # sector 109's first orbit, which ended 2026-10-18, was in TESScut by 2026-10-27
TIMING_STEP_D = 1 / 1440
MIN_GROUP_SNR = 5.0  # a sector's transits must stand out this much to time them
LOST_FRACTION = 0.25  # past this timing error, as a fraction of the period, the transit is lost
BASELINE_D = 0.5 / 24  # out-of-transit time wanted on each side from the ground
DARK_SUN_ALT = -12.0
MIN_ALT = 30.0
# Rough ground-photometry model for "how big a telescope?": photons and scintillation only, then
# a factor for the correlated noise real light curves carry. Tmag stands in for the I band.
PHOTONS_MAG0 = 1.5e9  # e-/s per m^2 of aperture at I = 0, after ~30% throughput
SCINT_CORRECTION = 1.5  # Young's scintillation formula underestimates it about this much (Osborn et al. 2015)
GROUND_RED_NOISE = 2.0
GROUND_AIRMASS, GROUND_HEIGHT_M = 1.3, 2000.0
APERTURES_M = (0.4, 1.0, 2.0)
SITES = {  # Las Cumbres Observatory: lat, lon (deg), height (m)
    "Siding Spring, Australia": (-31.273, 149.071, 1116),
    "SAAO, South Africa": (-32.380, 20.810, 1460),
    "Cerro Tololo, Chile": (-30.168, -70.805, 2198),
    "McDonald, Texas": (30.680, -104.015, 2027),
    "Teide, Tenerife": (28.300, -16.511, 2390),
    "Haleakala, Hawaii": (20.707, -156.258, 3055),
}


def mid_times(lc: dict, sig: dict, total: float, ingress: float, groups: np.ndarray) -> list[dict]:
    """One mid-transit time per group of data (a sector, or a single transit), as a shift from
    the search ephemeris at the group's mean epoch. The error is where chi2 rises by one,
    after scaling for red noise."""
    p, t0 = sig["period"], sig["t0"]
    beta = red_noise(lc, sig, np.ones(len(lc["t"]), bool))
    shifts = np.arange(-sig["dur"], sig["dur"] + TIMING_STEP_D / 2, TIMING_STEP_D)
    out = []
    for g in np.unique(groups):
        m = groups == g
        x = phase_offset(lc["t"][m], p, t0)
        near = np.abs(x) < total / 2 + sig["dur"] + 2 * TIMING_STEP_D
        inside = np.abs(x) < total / 2
        if inside.sum() < 3:
            continue
        x, y, w = x[near], lc["y"][m][near], lc["w"][m][near]
        s1, sy = w.sum(), np.sum(w * y)
        gain = np.zeros(len(shifts))
        for i, sh in enumerate(shifts):
            mdl = trapezoid(x - sh, total, ingress)
            sm, smm, smy = np.sum(w * mdl), np.sum(w * mdl * mdl), np.sum(w * mdl * y)
            cov, var = smy - sm * sy / s1, smm - sm * sm / s1
            if var > 0 and cov < 0:
                gain[i] = cov * cov / var
        best = int(np.argmax(gain))
        snr = float(np.sqrt(gain[best]) / beta)
        if snr < MIN_GROUP_SNR or best in (0, len(shifts) - 1):
            continue
        dchi2 = (gain[best] - gain) / beta**2
        lo, hi = best, best
        while lo > 0 and dchi2[lo - 1] <= 1:
            lo -= 1
        while hi < len(shifts) - 1 and dchi2[hi + 1] <= 1:
            hi += 1
        epochs = np.round((lc["t"][m] - t0) / p)[np.abs(phase_offset(lc["t"][m], p, t0)) < total / 2]
        wt = lc["w"][m][np.abs(phase_offset(lc["t"][m], p, t0)) < total / 2]
        out.append({"group": int(g), "epoch": float(np.sum(wt * epochs) / wt.sum()),
                    "transits": int(len(np.unique(epochs))), "shift_min": float(shifts[best] * 1440),
                    "err_min": float(max((shifts[hi] - shifts[lo]) / 2, TIMING_STEP_D) * 1440), "snr": snr})
    return out


def ephemeris(lc: dict, sig: dict) -> dict:
    """Period and mid-transit time with errors, from a straight line through the timed groups."""
    s = shape(lc, sig)
    total = s["t14_h"] / 24 if s["t14_h"] else sig["dur"]
    ingress = min(s["best_ingress_fraction"] or 0.15, 0.5)
    per = "sector"
    times = mid_times(lc, sig, total, ingress, lc["sector"])
    if len(times) < 2:  # one sector: time each transit instead
        per = "transit"
        times = mid_times(lc, sig, total, ingress, np.round((lc["t"] - sig["t0"]) / sig["period"]).astype(int))
    if len(times) < 2:
        return {"ok": False, "why": "fewer than two groups of transits could be timed"}
    n = np.array([r["epoch"] for r in times])
    y = np.array([r["shift_min"] for r in times]) / 1440
    e = np.array([r["err_min"] for r in times]) / 1440
    ref = float(np.round(np.sum(n / e**2) / np.sum(1 / e**2)))
    a = np.vstack([np.ones_like(n), n - ref]).T / e[:, None]
    cov = np.linalg.inv(a.T @ a)
    coef = cov @ a.T @ (y / e)
    chi2 = float(np.sum((a @ coef - y / e) ** 2))
    dof = len(n) - 2
    if dof > 0:
        cov *= max(1.0, chi2 / dof)
    t0 = sig["t0"] + ref * sig["period"] + coef[0]
    return {"ok": True, "period": sig["period"] + coef[1], "period_err": float(np.sqrt(cov[1, 1])),
            "t0_btjd": float(t0), "t0_err_min": float(np.sqrt(cov[0, 0]) * 1440), "cov_t0_p": float(cov[0, 1]),
            "t14_h": float(total * 24), "timed_per": per, "groups": len(times), "chi2": chi2, "dof": dof,
            "times": times}


def predict(eph: dict, t_btjd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mid-times (BTJD) of the transits nearest t_btjd and their errors (days)."""
    n = np.round((np.asarray(t_btjd) - eph["t0_btjd"]) / eph["period"])
    var = (eph["t0_err_min"] / 1440) ** 2 + n**2 * eph["period_err"] ** 2 + 2 * n * eph["cov_t0_p"]
    return eph["t0_btjd"] + n * eph["period"], np.sqrt(np.maximum(var, 0))


def iso(btjd: float) -> str:
    return Time(btjd + BTJD, format="jd", scale="tdb").utc.iso[:16]


def tess_sectors(tic: int, star: dict, eph: dict, now: float) -> list[dict]:
    """Sectors still to come (or under way) that put the star on a camera."""
    from tess_stars2px import TESS_Spacecraft_Pointing_Data, tess_stars2px_function_entry as stars2px
    pointing = TESS_Spacecraft_Pointing_Data()
    mids = dict(zip(np.asarray(pointing.sectors).tolist(), np.asarray(pointing.midtimes) - BTJD))
    _, _, _, sec, cam, ccd, *_ = stars2px([tic], [star["ra"]], [star["dec"]])
    out = []
    for s, c, d in zip(np.asarray(sec).tolist(), np.asarray(cam).tolist(), np.asarray(ccd).tolist()):
        if s < 0 or s not in mids or s > 1000:  # 1000+ are not survey sectors
            continue
        start, end = mids[s] - SECTOR_HALF_D, mids[s] + SECTOR_HALF_D
        if end < now:
            continue
        n0, n1 = np.ceil((start - eph["t0_btjd"]) / eph["period"]), np.floor((end - eph["t0_btjd"]) / eph["period"])
        _, err = predict(eph, np.array([mids[s]]))
        out.append({"sector": s, "camera": c, "ccd": d, "start": iso(start)[:10], "end": iso(end)[:10],
                    "public_from": iso(end + RELEASE_LAG_D)[:10], "transits": int(max(0, n1 - n0 + 1)),
                    "timing_err_min": round(float(err[0]) * 1440, 1),
                    "lost": bool(err[0] > LOST_FRACTION * eph["period"])})
    return sorted(out, key=lambda r: r["sector"])


def ground_windows(star: dict, eph: dict, now: float, days: float, sites: dict = SITES) -> list[dict]:
    """Transits in the next `days` that a site sees whole, with baseline, in the dark and above
    MIN_ALT. Times are UTC at the site: the barycentric light-travel time is taken off."""
    coord = SkyCoord(star["ra"], star["dec"], unit="deg")
    n = np.arange(np.ceil((now - eph["t0_btjd"]) / eph["period"]), np.floor((now + days - eph["t0_btjd"]) / eph["period"]) + 1)
    if not len(n):
        return []
    mid, err = predict(eph, eph["t0_btjd"] + n * eph["period"])
    half = eph["t14_h"] / 48 + err + BASELINE_D
    out = []
    for name, (lat, lon, h) in sites.items():
        loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=h * u.m)
        t = Time(mid + BTJD, format="jd", scale="tdb", location=loc)
        t = t - t.light_travel_time(coord, "barycentric")
        steps = np.linspace(-1, 1, 5)
        when = t[:, None] + (half[:, None] * steps[None, :]) * u.day
        frame = AltAz(obstime=when, location=loc)
        alt = coord.transform_to(frame).alt.deg
        sun = get_sun(when).transform_to(frame).alt.deg
        ok = (alt >= MIN_ALT).all(axis=1) & (sun <= DARK_SUN_ALT).all(axis=1)
        for i in np.flatnonzero(ok):
            out.append({"site": name, "mid_utc": t[i].utc.iso[:16],
                        "from_utc": (t[i] - half[i] * u.day).utc.iso[:16], "to_utc": (t[i] + half[i] * u.day).utc.iso[:16],
                        "timing_err_min": round(float(err[i]) * 1440, 1), "alt_mid": round(float(alt[i, 2]), 0)})
    return sorted(out, key=lambda r: r["mid_utc"])


def ground_snr(tmag: float, depth_ppm: float, t14_h: float, aperture_m: float) -> float:
    """Expected SNR of one full transit from the ground, with as much time out of transit as in."""
    area = np.pi * (aperture_m / 2) ** 2 * 0.8  # less the secondary mirror
    rate = PHOTONS_MAG0 * 10 ** (-0.4 * tmag) * area  # e-/s
    scint = (SCINT_CORRECTION * 0.09 * (aperture_m * 100) ** (-2 / 3) * GROUND_AIRMASS ** 1.75
             * np.exp(-GROUND_HEIGHT_M / 8000) / np.sqrt(2.0))  # fractional, for 1 s
    per_second = np.hypot(1 / np.sqrt(rate), scint)
    t_in = t14_h * 3600
    depth_err = per_second * np.sqrt(1 / t_in + 1 / t_in) * GROUND_RED_NOISE
    return float(depth_ppm * 1e-6 / depth_err)


def when(tic: int, sig: dict, lc_dir: Path, days: float = 90, now: Time | None = None) -> dict:
    raw = load_lc(tic, lc_dir)
    lc = detrend(raw["time"], raw["flux"], raw["sector"])
    eph = ephemeris(lc, sig)
    if not eph["ok"]:
        return {"tic": tic, "ephemeris": eph, "testable": False, "why": eph["why"]}
    star = star_info(tic)
    now_btjd = (now or Time.now()).tdb.jd - BTJD
    _, err_now = predict(eph, np.array([now_btjd]))
    tess = tess_sectors(tic, star, eph, now_btjd)
    ground = ground_windows(star, eph, now_btjd, days)
    lost = err_now[0] > LOST_FRACTION * eph["period"]
    next_tess = next((s for s in tess if s["transits"] and not s["lost"]), None)
    return {"tic": tic, "depth_ppm": sig["depth_ppm"], "ephemeris": eph,
            "timing_err_now_min": round(float(err_now[0]) * 1440, 1),
            "testable": bool(not lost and (next_tess or ground)),
            "next_tess": next_tess, "tess_sectors": tess,
            "ground_nights": len(ground), "ground": ground[:20],
            "ground_snr_per_transit": {f"{a:g} m": round(ground_snr(star["tmag"], sig["depth_ppm"], eph["t14_h"], a), 1)
                                       for a in APERTURES_M}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tic", type=int)
    ap.add_argument("--results", required=True, help="search results.csv with this star's signal")
    ap.add_argument("--lc", default=str(CACHE / "lc"), help="folder with the lc_NNN.npz shards")
    ap.add_argument("--days", type=float, default=90, help="ground windows over this many days")
    ap.add_argument("--now", help="pretend it is this UTC date (YYYY-MM-DD)")
    args = ap.parse_args()
    now = Time(args.now, scale="utc") if args.now else None
    out = when(args.tic, signal_from_results(args.tic, Path(args.results)), Path(args.lc), args.days, now)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
