"""A Zenodo preprint package for a candidate: the standard for every Tessa candidate.

From the candidate file (gate 2, novelty, ephemeris), its code checks, the validation tests
(pipeline/validate.py) and TRICERATOPS (pipeline/fpp.py, when it has run), it writes to
preprints/TIC<tic>/:
  preprint.html          the paper; PREPRINT_DATE and ZENODO_DOI stay as placeholders until the DOI is reserved
  fig_fold.png           the folded transit with the fitted model, and odd against even transits
  fig_sectors.png        depth sector by sector, and each season predicted from the others
  TIC<tic>_predicted_transits.csv   every transit to 2027-06-30: BJD_TDB, UTC, 1-sigma, and the
                         Las Cumbres sites that see it whole in the dark
  TIC<tic>_candidate.json           the parameters behind the paper
  zenodo_description.txt the description, keywords and related works to paste into Zenodo
and, with --pdf, renders TIC<tic>_preprint.pdf with headless Chrome (DOI and date filled from --doi/--date).

Run: python -m pipeline.preprint TIC [--pdf --doi 10.5281/zenodo.N --date 2026-10-30]
"""
import argparse
import html
import json
import math
import os
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

from .fpp import classify
from .paths import DATA, ROOT

OUT = ROOT / "preprints"
CHROME = Path(os.environ.get("CHROME") or Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe")
UNTIL = "2027-06-30"
AUTHOR = "Rushikesh Barve"
AFFIL = "Brave AI Solutions, Nashik, India &middot; rushi@braveai.solutions"
REPO = "https://github.com/Rishi943/tessa"
SITE = "https://tessa.braveai.solutions"
R_EARTH_PER_R_SUN = 109.08
# TRICERATOPS scenarios (Giacalone et al. 2021, Table 1); FPP = 1 - (TP + PTP + DTP)
SCENARIOS = {
    "TP": "a planet on the target", "EB": "the target an eclipsing binary", "EBx2P": "the same at twice the period",
    "PTP": "a planet on the target with an unresolved bound companion",
    "PEB": "the target an eclipsing binary with an unresolved bound companion", "PEBx2P": "the same at twice the period",
    "STP": "a planet on an unresolved bound companion", "SEB": "an unresolved bound companion that is an eclipsing binary",
    "SEBx2P": "the same at twice the period",
    "DTP": "a planet on the target, diluted by an unresolved background star",
    "DEB": "the target an eclipsing binary, diluted by an unresolved background star", "DEBx2P": "the same at twice the period",
    "BTP": "a planet on an unresolved background star", "BEB": "an unresolved background eclipsing binary",
    "BEBx2P": "the same at twice the period", "NTP": "a planet on a resolved neighbour",
    "NEB": "a resolved neighbour that is an eclipsing binary", "NEBx2P": "the same at twice the period",
}


def kind(teff: float) -> str:
    return "M" if teff < 3900 else "K" if teff < 5300 else "G" if teff < 6000 else "F"


def load(tic: int) -> dict:
    from .fpp import candidate
    from .tools import light_curve
    cand, vet = candidate(tic)
    checks, raw, lc = light_curve(tic)
    v = json.loads((DATA / "validation" / f"{tic}.json").read_text(encoding="utf-8"))
    fpp_path = DATA / "fpp" / f"fpp_{tic}.json"
    fpp = json.loads(fpp_path.read_text(encoding="utf-8")) if fpp_path.exists() else None
    return {"cand": cand, "checks": vet.get("checks", vet), "raw": raw, "lc": lc, "val": v, "fpp": fpp if fpp and "error" not in fpp else None}


# --- figures -----------------------------------------------------------------------------------

def figures(d: dict, folder: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .fit import model
    lc, eph = d["lc"], d["cand"]["when"]["ephemeris"]
    p, t0, t14 = eph["period"], eph["t0_btjd"], eph["t14_h"] / 24
    dt = (lc["t"] - t0 + p / 2) % p - p / 2
    near = np.abs(dt) < 3 * t14
    fit = d["val"]["fit"]["best"]
    edges = np.arange(-3 * t14, 3 * t14 + 1e-9, 20 / 1440)

    def binned(mask):
        idx = np.digitize(dt[mask], edges)
        x, yv, e = [], [], []
        for i in np.unique(idx):
            m = idx == i
            w = lc["w"][mask][m]
            x.append(np.sum(w * dt[mask][m]) / w.sum())
            yv.append(np.sum(w * lc["y"][mask][m]) / w.sum())
            e.append(1 / np.sqrt(w.sum()))
        return np.array(x), np.array(yv), np.array(e)

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].plot(dt[near] * 24, lc["y"][near] * 1e6, ".", ms=1.5, color="0.75", rasterized=True)
    x, yv, e = binned(near)
    ax[0].errorbar(x * 24, yv * 1e6, e * 1e6, fmt="o", ms=3, color="k", elinewidth=0.8)
    if fit.get("status") == "ok":
        tt = np.linspace(-3 * t14, 3 * t14, 600)
        mod = model(tt + t0, np.full_like(tt, 600 / 86400), p, t0 + fit["dt0_min"] / 1440, fit["k"], fit["b"], fit["ars"])
        ax[0].plot(tt * 24, mod * 1e6, color="#c0392b", lw=1.2)
    lim = max(3 * d["cand"]["signal"]["depth_ppm"], 4 * np.median(e) * 1e6)
    ax[0].set_ylim(-lim, lim / 2)
    ax[0].set_xlabel("hours from mid-transit")
    ax[0].set_ylabel("relative flux (ppm)")
    epoch = np.round((lc["t"] - t0) / p).astype(int)
    for parity, colour, label in ((0, "#2c3e50", "even"), (1, "#e67e22", "odd")):
        x, yv, e = binned(near & (epoch % 2 == parity))
        ax[1].errorbar(x * 24, yv * 1e6, e * 1e6, fmt="o", ms=2.5, color=colour, elinewidth=0.7, label=label)
    ax[1].set_ylim(-lim, lim / 2)
    ax[1].set_xlabel("hours from mid-transit")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "fig_fold.png", dpi=200)
    plt.close(fig)

    sec = d["checks"]["sectors"]["per_sector"]
    seasons = d["val"]["seasons"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.4))
    ax[0].errorbar(range(len(sec)), [s["depth_ppm"] for s in sec], [s["err_ppm"] for s in sec], fmt="o", color="k")
    ax[0].axhline(d["cand"]["signal"]["depth_ppm"], color="#c0392b", lw=1)
    ax[0].set_xticks(range(len(sec)), [str(s["sector"]) for s in sec])
    ax[0].set_xlabel("TESS sector")
    ax[0].set_ylabel("depth (ppm)")
    ok = [s for s in seasons if s.get("predicted_snr") is not None]
    ax[1].bar(range(len(ok)), [s["predicted_snr"] for s in ok], color="#2c3e50")
    ax[1].set_xticks(range(len(ok)), ["+".join(str(x) for x in s["hidden_sectors"]) for s in ok])
    ax[1].set_xlabel("hidden season (sectors)")
    ax[1].set_ylabel("SNR at predicted times")
    fig.tight_layout()
    fig.savefig(folder / "fig_sectors.png", dpi=200)
    plt.close(fig)


# --- predicted transits ------------------------------------------------------------------------

def transits(d: dict) -> list[dict]:
    from astropy.coordinates import EarthLocation, SkyCoord
    from astropy.time import Time
    from .when import BTJD, SITES, ground_windows, predict
    eph, star = d["cand"]["when"]["ephemeris"], d["cand"]["star"]
    now = Time.now().tdb.jd - BTJD
    end = Time(UNTIL).tdb.jd - BTJD
    n = np.arange(math.ceil((now - eph["t0_btjd"]) / eph["period"]), math.floor((end - eph["t0_btjd"]) / eph["period"]) + 1)
    mid, err = predict(eph, eph["t0_btjd"] + n * eph["period"])
    coord = SkyCoord(star["ra"], star["dec"], unit="deg")
    geo = EarthLocation.from_geocentric(0, 0, 0, unit="m")
    t = Time(mid + BTJD, format="jd", scale="tdb", location=geo)
    utc = (t - t.light_travel_time(coord, "barycentric")).utc
    windows = ground_windows(star, eph, now, end - now)
    rows = []
    for i in range(len(n)):
        row = {"epoch": int(n[i]), "t_mid_bjd_tdb": round(float(mid[i] + BTJD), 5), "t_mid_utc": utc[i].iso[:16],
               "sigma_min": round(float(err[i]) * 1440, 1), "duration_hr": round(eph["t14_h"], 2)}
        for site in SITES:
            hit = next((w for w in windows if w["site"] == site and abs(Time(w["mid_utc"]).jd - utc[i].jd) < 0.1), None)
            row[site] = f"{hit['from_utc'][11:]}-{hit['to_utc'][11:]} UTC" if hit else ""
        rows.append(row)
    return rows


# --- the paper ---------------------------------------------------------------------------------

CSS = """
@page { size: A4; margin: 22mm 20mm 20mm; }
body { margin: 0; background: #fff; color: #111; font: 10.4pt/1.32 Georgia, "Times New Roman", serif; }
h1 { margin: 0 0 8pt; font-size: 15.5pt; line-height: 1.22; text-align: center; letter-spacing: -0.1pt; }
h2 { margin: 12pt 0 5pt; font-size: 11pt; text-transform: uppercase; letter-spacing: 0.6pt; }
.who { margin: 0; text-align: center; font-size: 11pt; }
.affil, .date { margin: 1pt 0 0; text-align: center; font-size: 9.2pt; color: #333; }
.date { margin-bottom: 12pt; }
.abstract { margin: 0 10mm 12pt; padding: 6pt 0; border-top: 0.6pt solid #888; border-bottom: 0.6pt solid #888; font-size: 9.6pt; text-align: justify; }
p, li { margin: 0 0 6pt; text-align: justify; }
ul { margin: 2pt 0 6pt; padding-left: 14pt; }
table { margin: 5pt auto 8pt; border-collapse: collapse; font-size: 8.8pt; }
caption { margin-bottom: 2pt; caption-side: top; text-align: left; font-style: italic; font-size: 8.8pt; }
th, td { padding: 2pt 7pt 2pt 0; text-align: left; vertical-align: top; }
thead { border-top: 0.8pt solid #111; border-bottom: 0.5pt solid #111; }
tbody { border-bottom: 0.8pt solid #111; }
figure { margin: 8pt 0 10pt; page-break-inside: avoid; }
figure img { display: block; width: 100%; }
figcaption { margin-top: 4pt; font-size: 8.8pt; text-align: justify; color: #222; }
.refs p { margin: 0 0 2pt 12pt; text-indent: -12pt; font-size: 8.8pt; text-align: left; }
"""


def e(x) -> str:
    return html.escape(str(x))


def table(head: list[str], rows: list[list], caption: str = "") -> str:
    cap = f"<caption>{caption}</caption>" if caption else ""
    th = "".join(f"<th>{h}</th>" for h in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table>{cap}<thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def paper(d: dict, tr: list[dict]) -> tuple[str, str]:
    c, k, v, f = d["cand"], d["checks"], d["val"], d["fpp"]
    tic, star, sig, eph, when = c["tic"], c["star"], c["signal"], c["when"]["ephemeris"], c["when"]
    fit = v["fit"]
    rp = fit.get("rp_rearth") or {"median": c["size"]["rp_rearth"], "lo": None, "hi": None}
    rp_txt = f"{rp['median']:.2f}" + (f"<sup>+{rp['hi'] - rp['median']:.2f}</sup><sub>&minus;{rp['median'] - rp['lo']:.2f}</sub>" if rp["lo"] else "")
    sp = kind(star["teff"])
    oe, sec, sh, nb, cen = k["odd_even"], k["sectors"], k["shape"], k["neighbours"], k.get("centroid") or {}
    gaia, seasons, flip = v["gaia"], v["seasons"], v["flipped"]
    sectors = sorted({int(s) for s in np.unique(d["raw"]["sector"])})
    n_tr = k["refit"]["transits"]
    nxt = when.get("next_tess") or {}
    ok_seasons = [s for s in seasons if s.get("same_period")]
    rho = fit.get("rho_ratio")
    per_transit = sig["snr"] / math.sqrt(max(n_tr, 1))
    expect = per_transit * math.sqrt(nxt.get("transits", 0)) if nxt else None
    fclass = classify(f["FPP_mean"], f["NFPP_mean"]) if f else None
    passed = fclass in ("validated", "likely planet")
    near_open = bool(f) and not passed and f["FPP_mean"] < 0.5 and f["NFPP_mean"] >= 0.001

    title = (f"TIC {tic}.01: a {rp['median']:.1f} R<sub>&oplus;</sub> planet candidate on a {eph['period']:.2f}-day orbit "
             f"around a nearby {sp} dwarf, found in <i>TESS</i> full-frame images")
    tests = [
        f"odd and even transits agree ({oe['diff_sigma']:.1f}&sigma;) and there is no significant secondary eclipse (SNR {k['secondary']['snr']:.1f});",
        f"the depth is consistent across sectors (<i>p</i> = {sec['p_constant']:.2f});",
    ]
    if rho:
        tests.append(f"the stellar density implied by the transit shape matches the catalogue value (ratio {rho['median']:.2f}, 68% interval {rho['lo']:.2f}&ndash;{rho['hi']:.2f});")
    if cen.get("verdict") == "on_target":
        tests.append(f"difference imaging places the source on the target ({cen['offset_arcsec']:.1f}&Prime; &plusmn; {cen['err_arcsec']:.1f}&Prime;, {cen['sectors_used']} sectors) and on none of the {nb['could_cause']} Gaia neighbours bright enough to fake it;")
    if (gaia.get("ruwe") or 9) < 1.4:
        tests.append(f"Gaia DR3 astrometry (RUWE {gaia['ruwe']:.2f}){' and radial velocities' if gaia.get('rv_chisq_pvalue') is not None else ''} show no sign that the target is a binary;")
    if ok_seasons and len(ok_seasons) == len(seasons):
        tests.append(f"in a leave-one-season-out test, each of {len(seasons)} seasons is predicted from the others and shows the dip at the predicted times;")
    if flip.get("flipped_snr"):
        tests.append(f"the identical search on this star's upside-down light curve finds nothing stronger than SNR {flip['flipped_snr']:.1f}, against {flip['dip_snr']:.1f} for the dip;")
    if passed:  # a test passed; short of validation, the bullet says so
        tests.append(f"TRICERATOPS gives FPP = {f['FPP_mean']:.3f} &plusmn; {f['FPP_std']:.3f} and NFPP = {f['NFPP_mean']:.1e}"
                     + (", below the statistical-validation bar (FPP &lt; 0.015, NFPP &lt; 0.001)." if fclass == "validated"
                        else ": a likely planet (FPP &lt; 0.5 and NFPP &lt; 0.001; Giacalone et al. 2021), short of the statistical-validation bar (FPP &lt; 0.015, NFPP &lt; 0.001)."))
    fpp_note = ""
    if near_open:  # FPP alone looks good, but a nearby star isn't excluded: said outside the list of passed tests
        fpp_note = (f" TRICERATOPS gives FPP&nbsp;=&nbsp;{f['FPP_mean']:.3f}&nbsp;&plusmn;&nbsp;{f['FPP_std']:.3f}, but NFPP&nbsp;=&nbsp;{f['NFPP_mean']:.1e} "
                    "is above the 0.001 that the &ldquo;likely planet&rdquo; class also requires (Giacalone et al. 2021): by that test a nearby star "
                    "is not yet excluded as the source.")
    elif f and not passed:
        fpp_note = f" TRICERATOPS gives FPP&nbsp;=&nbsp;{f['FPP_mean']:.3f} and NFPP&nbsp;=&nbsp;{f['NFPP_mean']:.1e} ({fclass})."
    abstract = (
        f"We report a transiting planet candidate around TIC&nbsp;{tic} (Gaia DR3 {gaia['source_id']}), a <i>T</i>&nbsp;=&nbsp;{star['tmag']:.1f} "
        f"{sp} dwarf at {star['distance']:.1f}&nbsp;pc, found by Tessa, an open search of <i>TESS</i> full-frame-image light curves of nearby K and M dwarfs "
        f"that were mostly never observed at 2-minute cadence. Data from {len(sectors)} sectors show {n_tr} transits with period "
        f"<i>P</i>&nbsp;=&nbsp;{eph['period']:.6f}&nbsp;&plusmn;&nbsp;{eph['period_err']:.6f}&nbsp;d, depth {sig['depth_ppm']:.0f}&nbsp;ppm and duration "
        f"{eph['t14_h']:.2f}&nbsp;h (SNR&nbsp;{sig['snr']:.1f}), implying a planet radius of {rp_txt}&nbsp;R<sub>&oplus;</sub>. "
        + ("The signal passes every test available to us:" if passed or not f else "The signal passes these tests:")
        + "<ul>" + "".join(f"<li>{t[:-1] + '.' if i == len(tests) - 1 else t}</li>" for i, t in enumerate(tests)) + "</ul>"
        + fpp_note.strip() + (" " if fpp_note else "")
        + f"The candidate matches no known planet, TOI, CTOI, catalogued eclipsing binary or variable star. We give an ephemeris and predicted transit times "
        f"(timing uncertainty &asymp;{when['timing_err_now_min']:.0f}&nbsp;min)"
        + (f" for <i>TESS</i> Sector {nxt['sector']}, which observes the star from {nxt['start']} to {nxt['end']}," if nxt else "")
        + " and for ground-based observations, and set out in advance the criterion by which new data will test the prediction.")

    star_rows = [["TIC", tic], ["Gaia DR3", gaia["source_id"]], ["RA, Dec (J2000, deg)", f"{star['ra']:.5f}, {star['dec']:.5f}"],
                 ["<i>TESS</i> magnitude", f"{star['tmag']:.2f}"], ["<i>T</i><sub>eff</sub>", f"{star['teff']:.0f} K"],
                 ["Radius (TIC / Gaia FLAME)", f"{star['rad']:.3f} / {gaia['radius_flame']:.3f} R<sub>&#9737;</sub>" if gaia.get("radius_flame") else f"{star['rad']:.3f} R<sub>&#9737;</sub>"],
                 ["Mass", f"{star['mass']:.2f} M<sub>&#9737;</sub>"], ["Distance", f"{star['distance']:.1f} pc"],
                 ["Contamination ratio (TIC)", f"{star['contratio']:.3f}"], ["Gaia RUWE", f"{gaia['ruwe']:.2f}"]]
    if gaia.get("rv_chisq_pvalue") is not None:
        star_rows.append(["Gaia RV constancy <i>p</i>", f"{gaia['rv_chisq_pvalue']:.2f} ({gaia.get('rv_nb_transits') or 0:.0f} epochs)"])
    if gaia.get("ipd_frac_multi_peak") is not None:
        star_rows.append(["Gaia image doubling (ipd_frac_multi_peak)", f"{gaia['ipd_frac_multi_peak']:.0f}%"])

    def q(x, fmt="{:.3f}"):
        return f"{fmt.format(x['median'])} ({fmt.format(x['lo'])}&ndash;{fmt.format(x['hi'])})" if x else "&ndash;"
    sig_rows = [["Period", f"{eph['period']:.6f} &plusmn; {eph['period_err']:.6f} d"],
                ["Mid-transit time", f"BTJD {eph['t0_btjd']:.4f} &plusmn; {eph['t0_err_min']:.1f} min"],
                ["Depth (box)", f"{sig['depth_ppm']:.0f} ppm"], ["Duration, first to fourth contact", f"{eph['t14_h']:.2f} h"],
                ["Transits observed", f"{n_tr} in sectors {', '.join(map(str, sectors))}"], ["Box-search SNR", f"{sig['snr']:.1f}"],
                ["Radius ratio <i>R</i><sub>p</sub>/<i>R</i><sub>&#9733;</sub> (fit)", q(fit.get("k"), "{:.4f}")],
                ["Impact parameter (fit)", q(fit.get("b"), "{:.2f}")],
                ["Planet radius (fit errors only)", f"{rp_txt} R<sub>&oplus;</sub>"],
                ["Stellar density, fit / catalogue", q(rho, "{:.2f}")]]

    check_rows = [
        ["Odd vs even depth", f"{oe['depth_odd_ppm']:.0f} vs {oe['depth_even_ppm']:.0f} ppm, {oe['diff_sigma']:.1f}&sigma;", "no sign of an eclipsing binary at twice the period"],
        ["Secondary eclipse", f"SNR {k['secondary']['snr']:.1f} at phase {k['secondary']['phase']:.2f}; largest brightening SNR {k['secondary']['best_brightening_snr']:.1f}", "consistent with noise"],
        ["Depth by sector", ", ".join(f"{s['depth_ppm']}&plusmn;{s['err_ppm']}" for s in sec["per_sector"]) + f" ppm; &chi;<sup>2</sup> = {sec['chi2']:.2f} for {sec['dof']} dof", "steady" if sec["p_constant"] > 0.01 else "varies"],
        ["Shape", f"V shape disfavoured at {-sh['v_sigma']:.1f}&sigma;" if sh["v_sigma"] < 0 else f"V-like at {sh['v_sigma']:.1f}&sigma;", "transit-like" if sh["v_sigma"] < 0 else "possibly grazing"],
        ["Duration / central duration", f"{k['duration']['ratio']:.2f}", "consistent with a planet"],
        ["Variability at the period", f"{k['phase_curve']['amplitude_ppm']:.0f} ppm ({k['phase_curve']['ratio_to_depth']:.2f} of the depth)", "no variability mimicking the dip"],
        ["Gaia neighbours within 150&Prime;", f"{nb['gaia_stars_within']}; {nb['could_cause']} bright enough; {nb['could_cause_within_1px']} within one pixel ({nb['plausible_within_1px']} plausible)", "see centroid"],
    ]
    if cen:
        check_rows.append(["Difference-image centroid", f"{cen.get('offset_arcsec', 0):.1f}&Prime; &plusmn; {cen.get('err_arcsec', 0):.1f}&Prime; from the target ({cen.get('offset_sigma', 0):.1f}&sigma;), {cen.get('sectors_used', 0)} sectors", e(cen.get("why", ""))])
    check_rows.append(["Catalogues", "TOI, CTOI, NASA Exoplanet Archive, TESS EB catalogue, AAVSO VSX, Gaia DR3 eclipsing binaries, SIMBAD", "no match at this period or a simple harmonic"])

    season_rows = [[", ".join(map(str, s["hidden_sectors"])), f"{s['blind_period']:.5f}" if s.get("blind_period") else "&ndash;",
                    f"{s['blind_snr']:.1f}" if s.get("blind_snr") else "&ndash;",
                    f"{s['predicted_depth_ppm']} ppm, SNR {s['predicted_snr']}, {s['predicted_transits']} transits" if s.get("same_period") else "period not recovered"]
                   for s in seasons]
    fpp_par = ""
    if f:
        scen = sorted(f["scenarios_mean"].items(), key=lambda kv: -kv[1])[:5]
        fpp_par = (f"<p><b>TRICERATOPS.</b> With TRICERATOPS {e(f['triceratops'])} (Giacalone et al. 2021), {f['n_runs']} independent runs of "
                   f"{f['n_sim']:,} draws per scenario, " + (f"apertures from the TESS-SPOC sectors ({', '.join(map(str, f['geometry_sectors']))}; current TESScut cutouts no longer record their CCD origin, so it was taken from the same TESS-SPOC files)" if f.get('aperture', 'TESS-SPOC') == 'TESS-SPOC' else f"a circular {2:.0f}-pixel aperture (no TESS-SPOC sector exists for this star)") + " and the folded "
                   f"10-minute light curve of sectors {', '.join(map(str, f['lc_sectors']))}, we find FPP&nbsp;=&nbsp;{f['FPP_mean']:.3f}&nbsp;&plusmn;&nbsp;{f['FPP_std']:.3f} "
                   f"and NFPP&nbsp;=&nbsp;{f['NFPP_mean']:.1e}. The most probable scenarios are "
                   + "; ".join(f"{s}, {SCENARIOS.get(s, s)} ({p:.3f})" for s, p in scen) + ". "
                   + ("This meets the usual statistical-validation bar (FPP&nbsp;&lt;&nbsp;0.015, NFPP&nbsp;&lt;&nbsp;0.001)." if fclass == "validated"
                      else "This puts it in the &ldquo;likely planet&rdquo; class (FPP&nbsp;&lt;&nbsp;0.5 and NFPP&nbsp;&lt;&nbsp;0.001; Giacalone et al. 2021) but short of the validation bar (FPP&nbsp;&lt;&nbsp;0.015, NFPP&nbsp;&lt;&nbsp;0.001), so we do not claim validation." if fclass == "likely planet"
                      else "The FPP is below 0.5, but the NFPP is above 0.001, so the candidate is not in the &ldquo;likely planet&rdquo; class of Giacalone et al. (2021), which needs both, and we do not claim validation. TRICERATOPS does not use the difference-image centroid, which bears directly on the nearby stars." if near_open
                      else "The NFPP is above 0.1: TRICERATOPS favours a nearby false positive." if fclass == "likely nearby false positive"
                      else "This does not support a planet interpretation on its own.") + "</p>")
    else:
        fpp_par = "<p><b>TRICERATOPS.</b> Not yet run for this candidate.</p>"

    next_rows = [[r["epoch"], f"{r['t_mid_bjd_tdb'] - 2457000:.4f}", r["t_mid_utc"], f"{r['sigma_min']:.1f}"]
                 for r in tr if nxt and nxt["start"] <= r["t_mid_utc"][:10] <= nxt["end"]]
    grounds = when.get("ground") or []
    snr = when.get("ground_snr_per_transit") or {}
    rule = (f"<p><b>Test, set out in advance.</b> Fold the new data on this ephemeris without refitting and measure the box depth at the predicted times "
            f"(duration {eph['t14_h']:.2f}&nbsp;h). From the SNR per transit in the existing data ({per_transit:.1f}), "
            + (f"Sector {nxt['sector']}'s {nxt['transits']} transits should give SNR &asymp;&nbsp;{expect:.0f}. " if expect else "")
            + f"We will consider the prediction <b>confirmed</b> if the dip appears with SNR&nbsp;&ge;&nbsp;5 and a depth within 2&sigma; of {sig['depth_ppm']:.0f}&nbsp;ppm, "
            f"and <b>refuted</b> if the data reach the expected sensitivity and show SNR&nbsp;&lt;&nbsp;2 at the predicted times (within &plusmn;3&sigma; of the timing). "
            "A colour-dependent depth would point to a blended eclipsing binary. Becoming a planet needs repeated on-time transits and a mass from radial velocities.</p>")

    caveats = ["Full-frame images only; no 2-minute data exist for this star yet."]
    suspects = cen.get("suspects") or []
    plausible = [x for x in suspects if x["sep_arcsec"] < 21 and x["eclipse_needed"] <= 0.25]
    open_suspects = [x for x in suspects if x["sigma_from_centroid"] < 3 and x["eclipse_needed"] <= 1]
    if cen.get("sectors_used") and cen["sectors_used"] < len(sectors):
        caveats.append(f"The centroid uses {cen['sectors_used']} of {len(sectors)} sectors; the others' difference images were too weak.")
    if not cen:
        caveats.append("No difference-image centroid could be measured, so nearby stars are not excluded as the source.")
    elif open_suspects:
        caveats.append(f"{len(open_suspects)} nearby star(s) that could cause the dip lie within 3&sigma; of the measured centroid ("
                       + "; ".join(f"{x['sep_arcsec']:.0f}&Prime; away, would need a {x['eclipse_needed'] * 100:.1f}% eclipse" for x in open_suspects)
                       + "); multi-colour photometry or high-resolution imaging would settle it.")
    if f and fclass != "validated":
        caveats.append("TRICERATOPS does not validate the planet; high-resolution imaging would test the remaining scenarios.")
    if near_open:
        caveats.append(f"Its NFPP ({f['NFPP_mean']:.1e}) is above 0.001: ground-based photometry that resolves the nearby stars during a transit would test the nearby-star scenarios directly.")

    excluded = ""
    shut = [x for x in plausible if x["sigma_from_centroid"] >= 3]
    if shut:
        excluded = ("<p><b>Nearby stars.</b> " + "; ".join(
            f"The neighbour {x['sep_arcsec']:.0f}&Prime; away, which would need a {x['eclipse_needed'] * 100:.1f}% eclipse, lies {x['sigma_from_centroid']:.1f}&sigma; from the measured centroid"
            for x in shut) + ", so it is excluded as the source.</p>")
    figs = ("<figure><img src='fig_fold.png'><figcaption><b>Figure 1.</b> Left: the light curve folded on the ephemeris (grey: 10-minute points; black: 20-minute bins) "
            "with the fitted limb-darkened transit (red). Right: even and odd transits.</figcaption></figure>"
            "<figure><img src='fig_sectors.png'><figcaption><b>Figure 2.</b> Left: depth sector by sector (red: the box depth from all sectors). "
            "Right: the SNR of the dip in each season at the times predicted by a blind search of the other seasons.</figcaption></figure>")
    refs = ["Brown, T. M., et al. 2013, PASP, 125, 1031 (Las Cumbres Observatory)", "Caldwell, D. A., et al. 2020, RNAAS, 4, 201 (TESS-SPOC)",
            "Gaia Collaboration, Vallenari, A., et al. 2023, A&amp;A, 674, A1", "Giacalone, S., et al. 2021, AJ, 161, 24 (TRICERATOPS)",
            "Huang, C. X., et al. 2020, RNAAS, 4, 204 (QLP)", "Kov&aacute;cs, G., Zucker, S., &amp; Mazeh, T. 2002, A&amp;A, 391, 369",
            "Ricker, G. R., et al. 2015, JATIS, 1, 014003", "Stassun, K. G., et al. 2019, AJ, 158, 138 (TIC)"]

    body = f"""<h1>{title}</h1>
<p class="who">{AUTHOR}</p><p class="affil">{AFFIL}</p>
<p class="date">Preprint, not peer reviewed &middot; PREPRINT_DATE &middot; doi:ZENODO_DOI</p>
<div class="abstract"><b>Abstract.</b> {abstract}</div>
<h2>1. Search</h2>
<p>Tessa searches <i>TESS</i> full-frame-image light curves from TESS-SPOC (Caldwell et al. 2020) and QLP (Huang et al. 2020) for 99,461 K and M dwarfs within 100&nbsp;pc
from Gaia DR3 and the TIC (Stassun et al. 2019), most of them never 2-minute targets. Each light curve is detrended with a 0.5-day running median, cleaned of flares and
binned to 10 minutes, then searched with a box search (Kov&aacute;cs et al. 2002) over periods of 0.5&ndash;15 days on GPUs. The detection threshold is measured on the same
stars' light curves turned upside down, where no real dip exists: it is the level that at most 1% of flipped light curves pass. Signals that pass are vetted by code, by Meta's
Muse models, and by a second gate in code; only a signal no check can explain, that no catalogue lists and that predicts its next transits becomes a candidate. The code is open: {REPO}.</p>
{table(["", ""], sig_rows, "Table 1. The signal.")}
{figs}
<h2>2. The star</h2>
{table(["", ""], star_rows, "Table 2. TIC " + str(tic) + ".")}
<h2>3. Vetting</h2>
{table(["Check", "Result", "Meaning"], check_rows, "Table 3. Checks computed by code from the light curve and catalogues.")}
{excluded}
<p><b>Transit fit and stellar density.</b> A limb-darkened transit fitted to the 10-minute data (bootstrap over transits, {fit.get('n_boot', 0)} resamples) gives
<i>R</i><sub>p</sub>/<i>R</i><sub>&#9733;</sub>&nbsp;=&nbsp;{q(fit.get('k'), '{:.4f}')}, impact parameter {q(fit.get('b'), '{:.2f}')} and a stellar density
{q(rho, '{:.2f}')} times the catalogue value; fitting with the catalogue density fixed worsens &chi;<sup>2</sup> by {fit['best'].get('star_density_penalty', float('nan')):.1f}.
A blend or an eclipsing binary usually needs a very different density.</p>
<p><b>Leave one season out.</b> Each observing season is hidden in turn, the same blind search runs on the rest, and the period it finds predicts the hidden season.</p>
{table(["Hidden sectors", "Blind period (d)", "Blind SNR", "Dip at predicted times"], season_rows)}
{fpp_par}
<h2>4. Predictions</h2>
{table(["Epoch", "BTJD", "UTC (geocentric)", "&sigma; (min)"], next_rows, f"Table 4. Predicted transits in <i>TESS</i> Sector {nxt.get('sector')} (camera {nxt.get('camera')}, CCD {nxt.get('ccd')}); data public from about {nxt.get('public_from')}.") if next_rows else ""}
<p>Ground windows in the next 90 days at Las Cumbres Observatory sites (Brown et al. 2013), whole transit with the star above 30&deg; and the Sun below &minus;12&deg;:
{'; '.join(f"{e(g['site'])} {g['mid_utc']} UTC" for g in grounds[:8]) or 'none'}. Expected SNR for one transit:
{', '.join(f"{v} with {a}" for a, v in snr.items())}. Every predicted transit to {UNTIL}, with the sites that see it, is in the accompanying CSV.</p>
{rule}
<h2>5. Caveats</h2>
<ul>{''.join(f'<li>{x}</li>' for x in caveats)}</ul>
<h2>Data and code</h2>
<p>Search, vetting, tests and predictions: {REPO}. The candidate's case file, with the light curve and every check: {SITE}.
This work uses data from the <i>TESS</i> mission, funded by NASA's Science Mission Directorate and obtained from MAST at STScI, and from the ESA <i>Gaia</i> mission, processed by the Gaia DPAC.</p>
<div class="refs"><h2>References</h2>{''.join(f'<p>{r}</p>' for r in refs)}</div>"""
    page = f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>TIC {tic}.01</title><style>{CSS}</style></head><body>{body}</body></html>"
    import re
    plain = abstract.replace("<li>", "\n- ").replace("</ul>", "\n").replace("<ul>", "\n")
    plain = html.unescape(re.sub(r"<[^>]+>", "", plain))  # tags first: "&lt;" in the text must survive as "<"
    return page, plain


def write(tic: int) -> Path:
    d = load(tic)
    folder = OUT / f"TIC{tic}"
    folder.mkdir(parents=True, exist_ok=True)
    figures(d, folder)
    tr = transits(d)
    import csv
    with (folder / f"TIC{tic}_predicted_transits.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(tr[0]))
        w.writeheader()
        w.writerows(tr)
    c = d["cand"]
    params = {"name": f"TIC {tic}.01", "status": "planet candidate (not validated)", "host": {**c["star"], "gaia_dr3": d["val"]["gaia"]["source_id"]},
              "ephemeris": c["when"]["ephemeris"], "transit": {**c["signal"], **c["size"]}, "validation": d["val"], "triceratops": d["fpp"],
              "checks": {k: d["checks"].get(k) for k in ("odd_even", "secondary", "sectors", "shape", "duration", "phase_curve", "neighbours", "centroid")},
              "prepared": date.today().isoformat(), "preprint_doi": "ZENODO_DOI"}
    params["host"].pop("source_id", None)  # rounded in the vet files; gaia_dr3 is exact
    (folder / f"TIC{tic}_candidate.json").write_text(json.dumps(params, indent=1, default=float), encoding="utf-8")
    page, plain = paper(d, tr)
    (folder / "preprint.html").write_text(page, encoding="utf-8")
    eph = c["when"]["ephemeris"]
    desc = (f"TITLE\nTIC {tic}.01: a {(d['val']['fit'].get('rp_rearth') or {'median': c['size']['rp_rearth']})['median']:.1f} Earth-radius planet candidate on a "
            f"{eph['period']:.2f}-day orbit around a nearby {kind(c['star']['teff'])} dwarf, found in TESS full-frame images\n\n"
            f"DESCRIPTION\n{plain.strip()}\n\nFiles: the preprint (PDF); every predicted transit to {UNTIL}, with ground-site observability (CSV); "
            f"and the candidate parameters (JSON).\n\nKEYWORDS\nexoplanets\nTESS\ntransiting planet candidate\nTIC {tic}\n{kind(c['star']['teff'])} dwarf\n"
            f"transit photometry\n\nRELATED WORKS\nRelation: Is supplemented by\nIdentifier: {REPO}\nResource type: Software\n\n"
            "RESOURCE TYPE: Publication / Preprint\nCREATOR: Barve, Rushikesh (Brave AI Solutions)\nLICENCE: Creative Commons Attribution 4.0 International\n")
    (folder / "zenodo_description.txt").write_text(desc, encoding="utf-8")
    return folder


def render(folder: Path, tic: int, doi: str, when: str) -> Path:
    src = folder / "preprint.html"
    page = src.read_text(encoding="utf-8").replace("ZENODO_DOI", doi).replace("PREPRINT_DATE", when)
    out = folder / f"TIC{tic}_preprint.pdf"
    with tempfile.NamedTemporaryFile("w", suffix=".html", dir=folder, delete=False, encoding="utf-8") as f:
        f.write(page)
        tmp = Path(f.name)
    try:
        with tempfile.TemporaryDirectory() as profile:  # a throwaway profile: never the user's Chrome
            subprocess.run([str(CHROME), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                            f"--print-to-pdf={out}", tmp.as_uri()], check=True, capture_output=True, timeout=120)
    finally:
        tmp.unlink(missing_ok=True)
    if doi != "ZENODO_DOI":
        j = folder / f"TIC{tic}_candidate.json"
        j.write_text(j.read_text(encoding="utf-8").replace("ZENODO_DOI", doi), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tic", type=int)
    ap.add_argument("--pdf", action="store_true")
    ap.add_argument("--doi", default="ZENODO_DOI")
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()
    folder = write(args.tic)
    print(f"wrote {folder}")
    if args.pdf:
        print(f"wrote {render(folder, args.tic, args.doi, args.date)}")


if __name__ == "__main__":
    main()
