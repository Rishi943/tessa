"""Light curves for visitors to search in their own browser.

The held-back stars were set aside and never searched. Each one is detrended exactly as the search does it
(running median, flares dropped, 10-minute bins) and written as a small binary file the site's search reads:

  uint32 n, float64 t_ref, uint8 n_sectors,
  per sector: int16 sector, float32 1/noise^2, float32 cadence (days),
  then per point: uint32 seconds since t_ref, float16 relative flux, uint8 sector index, uint8 points binned

A bin's weight is points x 1/noise^2 and the time it covers is points x cadence, as in search.detrend. Times
round to the second and fluxes to float16 (about 3 significant digits), well inside the noise.

Also writes stars.json (where each star is and what it is) and parity files: the same format for survey stars
whose GPU search results are known, so the browser's search can be checked against the pipeline's.

Run: python -m pipeline.visitor [--limit 300]
"""
import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import CACHE, DATA, ROOT
from .search import BIN_D, detrend
from .vet import load_lc

OUT = DATA / "site" / "search"
PARITY = [240488148, 458191673]  # survey stars with known GPU results (a candidate and the rediscovery)


def encode(lc_raw: dict) -> bytes | None:
    """Detrend like the search, then pack the binned light curve."""
    t, flux, sector = lc_raw["time"], lc_raw["flux"], lc_raw["sector"]
    ok = np.isfinite(t) & np.isfinite(flux)
    lc = detrend(t[ok], flux[ok], sector[ok])
    if len(lc["t"]) < 100:
        return None
    sectors = np.unique(lc["sector"])
    # per sector: the inverse variance of one raw point and the raw cadence, recovered from the binned weights
    rows, count = [], np.zeros(len(lc["t"]), np.int64)
    for s in sectors:
        m = lc["sector"] == s
        cad = np.median(lc["span"][m] / np.maximum(np.round(lc["span"][m] / np.median(lc["span"][m])), 1))
        n = np.maximum(np.round(lc["span"][m] / cad), 1)
        count[m] = n
        rows.append((int(s), float(np.median(lc["w"][m] / n)), float(cad)))
    if count.max() > 255:
        return None
    t_ref = float(lc["t"].min())
    idx = np.searchsorted(sectors, lc["sector"])
    head = struct.pack("<IdB", len(lc["t"]), t_ref, len(sectors))
    head += b"".join(struct.pack("<hff", s, iv, cad) for s, iv, cad in rows)
    secs = np.round((lc["t"] - t_ref) * 86400).astype("<u4")
    return head + secs.tobytes() + lc["y"].astype("<f2").tobytes() + idx.astype("u1").tobytes() + count.astype("u1").tobytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=300, help="how many held-back stars to publish")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "parity").mkdir(exist_ok=True)
    held = [int(t) for t in (ROOT / "survey" / "holdout.txt").read_text(encoding="utf-8").split()]
    info = pd.read_parquet(DATA / "targets.parquet", columns=["tic", "rad", "mass", "ra", "dec", "parallax", "tmag", "teff"]
                           ).set_index("tic")
    stars, size = [], 0
    for tic in held:
        if len(stars) >= args.limit:
            break
        try:
            blob = encode(load_lc(tic, CACHE / "lc"))
        except (StopIteration, KeyError):
            continue
        if blob is None:
            continue
        (OUT / f"{tic}.bin").write_bytes(blob)
        size += len(blob)
        r = info.loc[tic]
        stars.append({"tic": tic, "ra": round(float(r.ra), 4), "dec": round(float(r.dec), 4), "dist": round(1000 / float(r.parallax), 2),  # Gaia, as on the map
                      "tmag": round(float(r.tmag), 2), "teff": None if pd.isna(r.teff) else int(r.teff),
                      "rad": None if pd.isna(r.rad) else round(float(r.rad), 4), "mass": None if pd.isna(r.mass) else round(float(r.mass), 4)})
    (OUT / "stars.json").write_text(json.dumps(stars, separators=(",", ":")), encoding="utf-8")
    print(f"{len(stars)} held-back stars, {size / 1e6:.1f} MB")

    # parity: the same format for survey stars, with the GPU search's own results to compare against
    with (DATA / "survey_results.csv").open(newline="", encoding="utf-8") as f:
        res = {int(r["tic"]): r for r in csv.DictReader(f)}
    stars_txt = pd.read_csv(DATA / "survey_stars.txt", sep=" ").set_index("tic")
    for tic in PARITY:
        (OUT / "parity" / f"{tic}.bin").write_bytes(encode(load_lc(tic, CACHE / "lc")))
        r = res[tic]
        (OUT / "parity" / f"{tic}.json").write_text(json.dumps({
            "tic": tic, "rad": float(stars_txt.loc[tic, "rad"]), "mass": float(stars_txt.loc[tic, "mass"]),
            "n_freq": int(r["n_freq"]), **{k: r[k] for k in ("dip_snr", "dip_period", "dip_transits", "bump_snr", "top_bump_snr", "gate1")}}),
            encoding="utf-8")
    print("parity files:", PARITY, "bin", BIN_D * 1440, "min")


if __name__ == "__main__":
    main()
