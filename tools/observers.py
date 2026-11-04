"""The candidates as an observer needs them: one table to plan a transit, and the same rows as CSV.

  python tools/observers.py      reads data/site/stars/<tic>.json (the export) and writes predictions/README.md and
                                 predictions/candidates.csv

Times are BJD_TDB (TESS BTJD + 2457000). The ephemeris is the refit over every timed group of transits
(pipeline/when.py); the timing error is propagated to the date of each window. Ground windows are the transit plus its
timing error, at night, with the star at least 30 degrees up, from the sites in pipeline/when.py.
"""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARS = ROOT / "data" / "site" / "stars"
OUT = ROOT / "predictions"
CANDIDATES = [9066956, 50099421, 61220826, 71503196, 79490453, 116046773, 166321785, 321290256]
SHOW = 3  # ground windows listed per candidate in the table


def ra_hms(ra: float) -> str:
    h = ra / 15
    return f"{int(h):02d}:{int(h * 60 % 60):02d}:{h * 3600 % 60:04.1f}"


def dec_dms(dec: float) -> str:
    s, a = "-" if dec < 0 else "+", abs(dec)
    return f"{s}{int(a):02d}:{int(a * 60 % 60):02d}:{a * 3600 % 60:04.1f}"


def row(tic: int, now: str) -> dict:
    d = json.loads((STARS / f"{tic}.json").read_text(encoding="utf-8"))
    assert d["status"] == "candidate", (tic, d["status"])
    c, w = d["checks"], d["when"]
    e, st = w["ephemeris"], c["star"]
    ground = [g for g in w["ground"] if g["from_utc"] >= now]
    tess = w.get("next_tess") or {}
    return {
        "tic": tic, "ra": round(st["ra"], 5), "dec": round(st["dec"], 5), "ra_hms": ra_hms(st["ra"]),
        "dec_dms": dec_dms(st["dec"]), "tmag": round(st["tmag"], 2), "teff_k": int(st["teff"]),
        "rstar_rsun": round(st["rad"], 3), "distance_pc": round(st["distance"], 1),
        "period_d": round(e["period"], 7), "period_err_d": float(f"{e['period_err']:.2g}"),
        "t0_bjd_tdb": round(e["t0_btjd"] + 2457000, 5), "t0_err_min": round(e["t0_err_min"], 1),
        "duration_h": round(e["t14_h"], 2), "depth_ppm": int(round(w["depth_ppm"])),
        "rp_rearth": round(c["size"]["rp_rearth"], 2), "tier": d["fpp"]["tier"],
        "fpp": float(f"{d['fpp']['FPP']:.2g}"), "nfpp": float(f"{d['fpp']['NFPP']:.2g}"),
        "timing_err_now_min": w["timing_err_now_min"], "ground_nights_listed": len(ground),
        "ground_nights_year": w["ground_nights"],
        "snr_per_transit_1m": w["ground_snr_per_transit"]["1 m"], "snr_per_transit_2m": w["ground_snr_per_transit"]["2 m"],
        "next_tess_sector": tess.get("sector"), "next_tess_dates": f"{tess['start']} to {tess['end']}" if tess else "",
        "next_tess_timing_err_min": tess.get("timing_err_min"),
        "next_ground": [f"{g['from_utc']} to {g['to_utc'][11:] if g['to_utc'][:10] == g['from_utc'][:10] else g['to_utc']} UTC, {g['site']} (alt {g['alt_mid']}°, ±{g['timing_err_min']} min)"
                        for g in ground[:SHOW]],
        "refuted_if": d["observing"]["refuted_if"].removeprefix("It is refuted if ").removeprefix("It would be refuted if "),
        "url": f"https://tessa.braveai.solutions/star/{tic}",
    }


def main() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    rows = sorted((row(t, now) for t in CANDIDATES), key=lambda r: (r["tier"] != "likely planet", r["ra"]))
    OUT.mkdir(exist_ok=True)
    flat = [k for k in rows[0] if k not in ("next_ground", "refuted_if")]
    with open(OUT / "candidates.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(flat + ["next_ground_windows"])
        for r in rows:
            wr.writerow([r[k] for k in flat] + [" | ".join(r["next_ground"])])
    L = ["# Tessa's candidates: predictions for observers", "",
         "Eight planet candidates from Tessa's survey of nearby K and M dwarfs in TESS full-frame images. None is",
         "confirmed: each is a repeating dip that every check Tessa runs could not explain. Each comes with a dated",
         "prediction of its next transits, so one night on a 1 to 2 m telescope can support it or rule it out.", "",
         f"Made {now} UTC by `python tools/observers.py` from the survey's export. The same rows, with every column, are in",
         "[candidates.csv](candidates.csv). Times are BJD_TDB; coordinates are J2000 (Gaia DR3 at epoch 2016).", "",
         "| TIC | RA, Dec | Tmag | Period (d) | T0 (BJD_TDB) | Duration | Depth | Size | TRICERATOPS | SNR per transit, 1 m / 2 m |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| [{r['tic']}]({r['url']}) | {r['ra_hms']} {r['dec_dms']} | {r['tmag']} | {r['period_d']} ± {r['period_err_d']} "
                 f"| {r['t0_bjd_tdb']} ± {r['t0_err_min']} min | {r['duration_h']} h | {r['depth_ppm']} ppm | {r['rp_rearth']} R⊕ "
                 f"| {r['tier']} (NFPP {r['nfpp']}) | {r['snr_per_transit_1m']} / {r['snr_per_transit_2m']} |")
    L += ["", "\"likely planet\" means TRICERATOPS gives NFPP under 0.001 (a nearby star is very unlikely to be the source);",
          "\"nearby star to rule out\" means FPP is under 0.5 but a nearby star is not yet ruled out, so high-resolution imaging",
          "or on-target photometry with a sharper camera is the first test.", "", "## Next windows", ""]
    for r in rows:
        L += [f"**TIC {r['tic']}** ({r['tier']}; timing error now ±{r['timing_err_now_min']} min; "
              f"{r['ground_nights_year']} usable ground nights in the coming year)", ""]
        L += [f"- {g}" for g in r["next_ground"]] or ["- no ground window in the list from now"]
        if r["next_tess_sector"]:
            L.append(f"- TESS sector {r['next_tess_sector']}, {r['next_tess_dates']} (±{r['next_tess_timing_err_min']} min)")
        L += [f"- Refuted if: {r['refuted_if']}", ""]
    L += ["Every window for the coming year is on each candidate's page on the site and in its case file",
          "(`data/vet/*_v10/candidates/<tic>.json`, field `when.ground`)."]
    (OUT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(rows)} candidates -> predictions/README.md, predictions/candidates.csv")


if __name__ == "__main__":
    main()
