"""Download TESS light curves for a list of stars and pack them into compact shards.

For each star and sector the best available product is kept:
SPOC 2-minute > TESS-SPOC FFI > QLP FFI. Flagged cadences are dropped and each sector
is normalised to its median, so a shard holds only what the search needs.

Self-contained (no package imports) so it can run as a Kaggle job:
  python pipeline/fetch.py --targets survey_order.txt --out /kaggle/working --max-minutes 660

Output: lc_NNN.npz shards of up to --chunk stars, each with arrays
  {tic}_time (BTJD, float64), {tic}_flux, {tic}_err (float32),
  {tic}_sector (int16), {tic}_source (int8: 0 SPOC, 1 TESS-SPOC, 2 QLP)
and manifest.csv with one row per star. Existing shards are skipped, so a rerun resumes.
"""
import argparse
import csv
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from astropy.io import fits

SOURCES = {"SPOC": 0, "TESS-SPOC": 1, "QLP": 2}  # lower is preferred
# QLP renamed its detrended flux from KSPSAP_FLUX to DET_FLUX in later sectors.
FLUX_COLUMNS = {"SPOC": ["PDCSAP_FLUX"], "TESS-SPOC": ["PDCSAP_FLUX"], "QLP": ["KSPSAP_FLUX", "DET_FLUX"]}
DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri={uri}"
T0 = time.time()


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min  {msg}", flush=True)


def is_light_curve(name: str) -> bool:
    # SPOC 2-min and TESS-SPOC end in _lc.fits, QLP in _llc.fits; skip 20-s fast-lc and pixel files.
    return (name.endswith("_lc.fits") and "fast" not in name) or name.endswith("_llc.fits")


def best_products(tics: list[int], sources: list[str]) -> dict[int, dict[int, tuple[str, str]]]:
    """For each star: {sector: (source, dataURI)}, keeping the preferred source per sector."""
    from astroquery.mast import Observations

    obs = Observations.query_criteria(
        target_name=[str(t) for t in tics], obs_collection=["HLSP", "TESS"],
        dataproduct_type="timeseries", provenance_name=sources,
    )
    if len(obs) == 0:
        return {}
    source_of = {str(o["obsid"]): str(o["provenance_name"]) for o in obs}
    tic_of = {str(o["obsid"]): int(o["target_name"]) for o in obs}
    sector_of = {str(o["obsid"]): int(o["sequence_number"]) for o in obs}

    best: dict[int, dict[int, tuple[str, str]]] = {}
    for p in Observations.get_product_list(obs):
        name = str(p["productFilename"])
        obsid = str(p["parent_obsid"])
        if not is_light_curve(name) or obsid not in source_of:
            continue
        tic, sector, source = tic_of[obsid], sector_of[obsid], source_of[obsid]
        current = best.setdefault(tic, {}).get(sector)
        if current is None or SOURCES[source] < SOURCES[current[0]]:
            best[tic][sector] = (source, str(p["dataURI"]))
    return best


def download(session: requests.Session, uri: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            r = session.get(DOWNLOAD_URL.format(uri=uri), timeout=120)
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def read_sector(data: bytes, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with fits.open(io.BytesIO(data)) as hdul:
        d = hdul[1].data
        column = next(c for c in FLUX_COLUMNS[source] if c in d.columns.names)
        t = np.asarray(d["TIME"], dtype=np.float64)
        f = np.asarray(d[column], dtype=np.float64)
        e = np.asarray(d[column + "_ERR"], dtype=np.float64)
        q = np.asarray(d["QUALITY"])
    ok = (q == 0) & np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
    t, f, e = t[ok], f[ok], e[ok]
    if len(f) == 0:
        return t, f, e
    med = np.median(f)
    return t, f / med, e / med


def fetch_chunk(tics: list[int], workers: int, sources: list[str]) -> tuple[dict[str, np.ndarray], list[dict]]:
    products = best_products(tics, sources)
    jobs = [(tic, sector, source, uri)
            for tic, sectors in products.items() for sector, (source, uri) in sectors.items()]

    session = requests.Session()

    def get(job):
        tic, sector, source, uri = job
        try:
            return job, read_sector(download(session, uri), source), None
        except Exception as e:
            return job, None, f"{e.__class__.__name__}: {e}"[:200]

    per_star: dict[int, list] = {t: [] for t in tics}
    errors: dict[int, list[str]] = {t: [] for t in tics}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (tic, sector, source, _), arrays, err in pool.map(get, jobs):
            if err:
                errors[tic].append(f"s{sector}: {err}")
            elif len(arrays[0]):
                per_star[tic].append((sector, SOURCES[source], *arrays))

    shard: dict[str, np.ndarray] = {}
    rows = []
    for tic in tics:
        parts = sorted(per_star[tic], key=lambda p: p[0])
        row = {"tic": tic, "n_sectors": len(parts), "n_points": 0, "sectors": "",
               "sources": "", "errors": "; ".join(errors[tic])}
        if parts:
            shard[f"{tic}_time"] = np.concatenate([p[2] for p in parts])
            shard[f"{tic}_flux"] = np.concatenate([p[3] for p in parts]).astype(np.float32)
            shard[f"{tic}_err"] = np.concatenate([p[4] for p in parts]).astype(np.float32)
            shard[f"{tic}_sector"] = np.concatenate([np.full(len(p[2]), p[0], np.int16) for p in parts])
            shard[f"{tic}_source"] = np.concatenate([np.full(len(p[2]), p[1], np.int8) for p in parts])
            row.update(n_points=len(shard[f"{tic}_time"]),
                       sectors=" ".join(str(p[0]) for p in parts),
                       sources=" ".join(str(p[1]) for p in parts))
        rows.append(row)
    return shard, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targets", required=True, help="text file, one TIC ID per line, in survey order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="only the first N stars")
    ap.add_argument("--chunk", type=int, default=100, help="stars per shard and per MAST query")
    ap.add_argument("--workers", type=int, default=6, help="parallel downloads")
    ap.add_argument("--max-minutes", type=float, default=660, help="stop cleanly after this long")
    ap.add_argument("--ffi-only", action="store_true",
                    help="TESS-SPOC and QLP only, like the never-2-minute survey stars (fine-tuning inputs)")
    args = ap.parse_args()
    sources = [s for s in SOURCES if not (args.ffi_only and s == "SPOC")]

    tics = [int(line) for line in Path(args.targets).read_text().split() if line.strip()]
    if args.limit:
        tics = tics[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.csv"
    fields = ["shard", "tic", "n_sectors", "n_points", "sectors", "sources", "errors"]
    if not manifest.exists():
        with manifest.open("w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    chunks = [tics[i:i + args.chunk] for i in range(0, len(tics), args.chunk)]
    log(f"{len(tics)} stars in {len(chunks)} shards")
    for n, chunk in enumerate(chunks):
        path = out / f"lc_{n:03d}.npz"
        if path.exists():
            continue
        if (time.time() - T0) / 60 > args.max_minutes:
            log(f"time limit reached; stopped before shard {n}")
            break
        for attempt in range(3):
            try:
                shard, rows = fetch_chunk(chunk, args.workers, sources)
                break
            except Exception as e:  # MAST query failures: wait and retry the whole chunk
                log(f"shard {n} attempt {attempt + 1} failed: {e.__class__.__name__}: {e}"[:300])
                time.sleep(60)
        else:
            continue
        tmp = out / f"lc_{n:03d}.tmp.npz"
        np.savez_compressed(tmp, **shard)
        tmp.replace(path)
        with manifest.open("a", newline="") as f:
            w = csv.DictWriter(f, fields)
            for row in rows:
                w.writerow({"shard": n, **row})
        got = sum(1 for r in rows if r["n_sectors"])
        mb = path.stat().st_size / 1e6
        log(f"shard {n}: {got}/{len(chunk)} stars with data, {mb:.0f} MB")
    log("done")


if __name__ == "__main__":
    main()
