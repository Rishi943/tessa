"""Build the ranked target pool: nearby K and M dwarfs with multi-year TESS coverage.

Funnel:
  Gaia DR3 stars within 100 pc with K/M colours
  -> matched to the TESS Input Catalog (via Gaia DR2 IDs; TIC 8.2 is built on DR2)
  -> dwarfs: 2300-5300 K, radius <= 0.9 Rsun, Tmag <= 14.5, contamination < 0.2
  -> novelty check A: not a TOI, CTOI, confirmed-planet host or known eclipsing binary
  -> observed in >= 1 TESS sector with released data
  -> ranked by the expected SNR of a 1.5 Earth-radius planet on a 10-day orbit

Run: python -m pipeline.targets [--trace TIC ...]
Writes data/targets.parquet and data/targets_funnel.json.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import catalogs
from .paths import CACHE, DATA

MAX_DISTANCE_PC = 100
MAX_GAIA_G = 15.5
MIN_BP_RP = 0.9  # roughly K0 and cooler
TEFF_RANGE = (2300, 5300)
MAX_RADIUS = 0.9
MAX_TMAG = 14.5  # the G cut ends about here for M dwarfs; 97% of a 1,000-star sample at 13.5-14.5
# had QLP or TESS-SPOC light curves (2026-10-27)
MAX_CONTRATIO = 0.2
SURVEY_FETCH = 6000
BATCH2_FETCH = 18000  # the next never-2-minute stars in rank order, after the pool reached Tmag 14.5
MIN_SECTORS = 1  # one-sector stars stay in the pool; the expected-SNR ranking pushes them down

# Latest sectors with released light curves on MAST, checked 2026-10-26:
# SPOC 2-min through 107, QLP FFI through 104.
LAST_2MIN_SECTOR = 107
LAST_FFI_SECTOR = 104
SECTOR_DAYS = 27.4
USEFUL_DAYS_PER_SECTOR = 23.0  # after gaps and quality masking

R_EARTH_RSUN = 0.009168
RSUN_PER_AU = 215.03

GAIA_FROM = f"""
FROM gaiadr3.gaia_source AS g
JOIN gaiadr3.dr2_neighbourhood AS n ON n.dr3_source_id = g.source_id
WHERE g.parallax >= {1000 / MAX_DISTANCE_PC}
  AND g.parallax_over_error >= 10
  AND g.bp_rp >= {MIN_BP_RP}
  AND g.phot_g_mean_mag <= {MAX_GAIA_G}
  AND g.dec >= {{dec_lo}} AND g.dec < {{dec_hi}}
"""
GAIA_COLUMNS = """
SELECT g.source_id, g.ra, g.dec, g.pmra, g.pmdec, g.parallax, g.phot_g_mean_mag, g.bp_rp,
       g.ruwe, g.phot_variable_flag, g.non_single_star,
       n.dr2_source_id, n.angular_distance
"""
# The Gaia archive silently truncates large downloads (~25k rows seen on 2026-10-26),
# so the sky is fetched in declination bands and each band is checked against a COUNT.
GAIA_DEC_BAND = 10

TIC_COLUMNS = ["ID", "GAIA", "Tmag", "Teff", "logg", "rad", "mass", "d", "contratio",
               "lumclass", "disposition", "ra", "dec"]


def gaia_nearby() -> pd.DataFrame:
    path = CACHE / "gaia_nearby.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from astroquery.gaia import Gaia

    Gaia.ROW_LIMIT = -1

    def run(query: str, attempts: int = 4):
        for attempt in range(attempts):
            try:
                return Gaia.launch_job_async(query).get_results()
            except Exception as e:  # the archive returns sporadic HTTP 500s
                if attempt == attempts - 1:
                    raise
                print(f"  Gaia error ({e.__class__.__name__}), retrying in 30 s", flush=True)
                time.sleep(30)

    band_dir = CACHE / "gaia_bands"
    band_dir.mkdir(exist_ok=True)
    frames = []
    for dec_lo in range(-90, 90, GAIA_DEC_BAND):
        part = band_dir / f"dec{dec_lo:+03d}.parquet"
        if not part.exists():
            where = GAIA_FROM.format(dec_lo=dec_lo, dec_hi=dec_lo + GAIA_DEC_BAND)
            expected = int(run("SELECT COUNT(*) AS n" + where)[0][0])
            band = run(GAIA_COLUMNS + where).to_pandas()
            if len(band) != expected:
                raise RuntimeError(f"Gaia band {dec_lo}..{dec_lo + GAIA_DEC_BAND}: got {len(band)} rows, expected {expected}")
            band.to_parquet(part)
            print(f"  Gaia dec {dec_lo:+} to {dec_lo + GAIA_DEC_BAND:+}: {expected:,} rows", flush=True)
        frames.append(pd.read_parquet(part))
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.lower() for c in df.columns]
    # A DR3 source can have several DR2 neighbours; keep the closest.
    df = df.sort_values("angular_distance").drop_duplicates("source_id")
    df.to_parquet(path)
    return df


def tic_lookup(dr2_ids: np.ndarray, batch: int = 2000, workers: int = 4) -> pd.DataFrame:
    path = CACHE / "tic.parquet"
    if path.exists():
        return pd.read_parquet(path)
    from astroquery.mast import Catalogs

    # Batches are cached individually so an interrupted lookup resumes where it stopped.
    batch_dir = CACHE / "tic_batches"
    batch_dir.mkdir(exist_ok=True)
    dr2_ids = np.sort(dr2_ids)
    parts = [(i, batch_dir / f"{i:07d}_{len(dr2_ids)}.parquet") for i in range(0, len(dr2_ids), batch)]
    todo = [(i, part) for i, part in parts if not part.exists()]

    def fetch(job):
        i, part = job
        ids = [str(x) for x in dr2_ids[i:i + batch]]
        t = Catalogs.query_criteria(catalog="Tic", GAIA=ids)
        tmp = part.with_suffix(".tmp")
        t[TIC_COLUMNS].to_pandas().to_parquet(tmp)
        tmp.replace(part)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, _ in enumerate(pool.map(fetch, todo), 1):
            print(f"  TIC lookup batch {len(parts) - len(todo) + n}/{len(parts)}", flush=True)
    df = pd.concat([pd.read_parquet(part) for _, part in parts], ignore_index=True)
    df = df.rename(columns={"ID": "tic", "GAIA": "dr2_source_id"})
    df["tic"] = pd.to_numeric(df["tic"]).astype("int64")
    df["dr2_source_id"] = pd.to_numeric(df["dr2_source_id"]).astype("int64")
    df.to_parquet(path)
    return df


def tess_sectors(df: pd.DataFrame, chunk: int = 20000) -> pd.DataFrame:
    """Every (tic, sector) pair that lands on a TESS CCD, past and scheduled."""
    path = CACHE / "sectors.parquet"
    queried_path = CACHE / "sectors_queried.npy"
    if path.exists() and queried_path.exists():
        if set(df["tic"]) <= set(np.load(queried_path).tolist()):
            return pd.read_parquet(path)
    from tess_stars2px import tess_stars2px_function_entry as stars2px

    frames = []
    for i in range(0, len(df), chunk):
        part = df.iloc[i:i + chunk]
        out = stars2px(part["tic"].to_numpy(), part["ra"].to_numpy(), part["dec"].to_numpy())
        frames.append(pd.DataFrame({"tic": out[0], "sector": out[3]}))
        print(f"  pointing {min(i + chunk, len(df))}/{len(df)}", flush=True)
    sectors = pd.concat(frames, ignore_index=True).drop_duplicates()
    sectors = sectors[sectors["tic"] >= 0]
    sectors.to_parquet(path)
    np.save(queried_path, df["tic"].to_numpy())
    return sectors


def noise_ppm_1hr(tmag: np.ndarray) -> np.ndarray:
    """Rough 1-hour photometric noise for an FFI light curve (ranking only, not for vetting).

    Photon noise from ~15,000 e-/s at Tmag 10, background from a ~10-pixel aperture,
    plus a 60 ppm systematics floor.
    """
    star = 15000 * 10 ** (-0.4 * (tmag - 10)) * 3600
    background = 10 * 100 * 3600
    photon = np.sqrt(star + background) / star * 1e6
    return np.hypot(photon, 60)


def expected_snr(df: pd.DataFrame, radius_earth: float = 1.5, period: float = 10.0) -> pd.Series:
    rs = df["rad"]
    ms = df["mass"].fillna(rs)  # mass ~ radius for M dwarfs when the TIC has no mass
    depth = (radius_earth * R_EARTH_RSUN / rs) ** 2 / (1 + df["contratio"].fillna(0))
    a_rs = (ms * (period / 365.25) ** 2) ** (1 / 3) * RSUN_PER_AU / rs
    duration_hr = period * 24 / np.pi / a_rs
    n_transits = df["n_sectors"] * USEFUL_DAYS_PER_SECTOR / period
    return depth * 1e6 / noise_ppm_1hr(df["tmag"]) * np.sqrt(n_transits * duration_hr)


def build(trace: list[int]) -> pd.DataFrame:
    funnel = []

    def step(label: str, frame: pd.DataFrame) -> pd.DataFrame:
        funnel.append({"step": label, "stars": len(frame)})
        print(f"{len(frame):>8,}  {label}", flush=True)
        for tic in list(trace):
            if "tic" in frame and tic not in set(frame["tic"]):
                print(f"          TIC {tic} dropped here", flush=True)
                trace.remove(tic)
        return frame

    gaia = step(f"Gaia DR3, <= {MAX_DISTANCE_PC} pc, BP-RP >= {MIN_BP_RP}, G <= {MAX_GAIA_G}", gaia_nearby())

    tic = tic_lookup(gaia["dr2_source_id"].to_numpy())
    tic = tic[tic["disposition"].isna()]  # drop TIC entries flagged SPLIT, DUPLICATE, ARTIFACT
    df = gaia.drop(columns=["ra", "dec"]).merge(tic, on="dr2_source_id", how="inner")
    df = df.drop_duplicates("tic").rename(columns={"Tmag": "tmag", "Teff": "teff", "d": "distance"})
    df = step("matched to the TESS Input Catalog", df)

    df = step("dwarf, K or M", df[
        (df["lumclass"] == "DWARF")
        & df["teff"].between(*TEFF_RANGE)
        & (df["rad"] <= MAX_RADIUS)
    ])
    df = step(f"Tmag <= {MAX_TMAG}", df[df["tmag"] <= MAX_TMAG])
    df = step(f"contamination < {MAX_CONTRATIO}", df[~(df["contratio"] >= MAX_CONTRATIO)])

    known = {
        "toi": catalogs.toi(),
        "ctoi": catalogs.ctoi(),
        "confirmed": catalogs.confirmed(),
        "eclipsing_binary": catalogs.tess_ebs(),
    }
    for name, table in known.items():
        df = step(f"not a {name.replace('_', ' ')} host", df[~df["tic"].isin(table["tic"])])

    sectors = tess_sectors(df)
    observed = sectors[sectors["sector"] <= LAST_FFI_SECTOR]
    per_star = observed.groupby("tic")["sector"].agg(["count", "min", "max"])
    df = df.merge(per_star, left_on="tic", right_index=True, how="left")
    df = df.rename(columns={"count": "n_sectors", "min": "first_sector", "max": "last_sector"})
    df["n_sectors"] = df["n_sectors"].fillna(0).astype(int)
    df["span_years"] = (df["last_sector"] - df["first_sector"]) * SECTOR_DAYS / 365.25
    upcoming = sectors[sectors["sector"] > LAST_FFI_SECTOR].groupby("tic")["sector"].apply(list)
    df["upcoming_sectors"] = df["tic"].map(upcoming).apply(lambda s: s if isinstance(s, list) else [])
    df = step(f"observed in >= {MIN_SECTORS} sector(s) (through sector {LAST_FFI_SECTOR})",
              df[df["n_sectors"] >= MIN_SECTORS])

    two_min = catalogs.two_min_targets(LAST_2MIN_SECTOR)
    df["two_min_sectors"] = df["tic"].map(two_min.groupby("tic").size()).fillna(0).astype(int)

    df["expected_snr"] = expected_snr(df)
    df = df.sort_values("expected_snr", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)

    ffi_only = int((df["two_min_sectors"] == 0).sum())
    print(f"{ffi_only:>8,}  of these were never 2-minute targets", flush=True)
    funnel.append({"step": "never 2-minute targets", "stars": ffi_only})

    # Survey order: stars never given 2-minute cadence (the least-searched), best expected SNR first.
    write_survey(df)

    df.to_parquet(DATA / "targets.parquet")
    (DATA / "targets_funnel.json").write_text(json.dumps(funnel, indent=2), encoding="utf-8")
    return df


def write_survey(df: pd.DataFrame) -> None:
    """survey_order.txt (TIC IDs, for the fetch) and survey_stars.txt (radius and mass, for the
    search's expected transit durations), both plain text so a Kaggle job can carry them.

    The first survey's SURVEY_FETCH stars are fixed once written, since the held-back set is drawn
    from them; batch2 takes the next BATCH2_FETCH never-2-minute stars in rank order."""
    never = df[df["two_min_sectors"] == 0]

    def write(stars: pd.DataFrame, name: str) -> None:
        (DATA / f"{name}_order.txt").write_text("\n".join(map(str, stars["tic"])) + "\n", encoding="utf-8")
        stars[["tic", "rad", "mass"]].to_csv(DATA / f"{name}_stars.txt", sep=" ", index=False, float_format="%.4f")

    if not (DATA / "survey_order.txt").exists():
        write(never.head(SURVEY_FETCH), "survey")
    first = {int(t) for t in (DATA / "survey_order.txt").read_text(encoding="utf-8").split()}
    write(never[~never["tic"].isin(first)].head(BATCH2_FETCH), "batch2")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", type=int, nargs="*", default=[], help="TIC IDs to follow through the funnel")
    args = ap.parse_args()
    df = build(list(args.trace))
    for tic in args.trace:
        row = df[df["tic"] == tic]
        if len(row):
            r = row.iloc[0]
            print(f"TIC {tic}: rank {r['rank']:,}, {r['n_sectors']} sectors, "
                  f"expected SNR {r['expected_snr']:.1f}, 2-min sectors {r['two_min_sectors']}")


if __name__ == "__main__":
    main()
