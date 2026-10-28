"""Star info for the TOI hosts, with the same fields pipeline.vet.star_info gives survey stars.

TIC 8.2 (MAST) for size, mass, distance, brightness and contamination; Gaia DR3 through DR2's
neighbourhood table (the TIC is built on DR2) for the DR3 ID, RUWE, variability and
non-single-star flags. Batches are cached, so a rerun resumes.

Run: python -m finetune.stars
Writes data/finetune/stars.json (TIC -> star dict).
"""
import json
import math
import time

import numpy as np
import pandas as pd

from pipeline.paths import CACHE, DATA

FOLDER = DATA / "finetune"
BATCH = 500
TIC_COLUMNS = ["ID", "GAIA", "Tmag", "Teff", "rad", "mass", "d", "contratio", "ra", "dec"]


def retry(fn, attempts=4):
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:  # MAST and the Gaia archive both return sporadic 500s
            if attempt == attempts - 1:
                raise
            print(f"  {e.__class__.__name__}, retrying in 20 s", flush=True)
            time.sleep(20)


def tic_rows(tics: list[int]) -> pd.DataFrame:
    from astroquery.mast import Catalogs
    folder = CACHE / "ft_tic"
    folder.mkdir(exist_ok=True)
    frames = []
    for i in range(0, len(tics), BATCH):
        part = folder / f"{i:05d}_{len(tics)}.parquet"
        if not part.exists():
            ids = [str(t) for t in tics[i:i + BATCH]]
            t = retry(lambda: Catalogs.query_criteria(catalog="Tic", ID=ids))
            t[TIC_COLUMNS].to_pandas().to_parquet(part)
            print(f"  TIC {min(i + BATCH, len(tics))}/{len(tics)}", flush=True)
        frames.append(pd.read_parquet(part))
    df = pd.concat(frames, ignore_index=True)
    df["tic"] = pd.to_numeric(df.pop("ID")).astype("int64")
    # 19-digit IDs: through float (pd.to_numeric with gaps) they lose their last digits
    gaia = [str(x).strip() if x is not None and not pd.isna(x) else "" for x in df.pop("GAIA")]
    df["dr2_source_id"] = pd.array([int(x) if x.isdigit() else None for x in gaia], dtype="Int64")
    return df.drop_duplicates("tic")


def gaia_rows(dr2_ids: list[int]) -> pd.DataFrame:
    from astroquery.gaia import Gaia
    folder = CACHE / "ft_gaia"
    folder.mkdir(exist_ok=True)
    frames = []
    for i in range(0, len(dr2_ids), BATCH):
        part = folder / f"{i:05d}_{len(dr2_ids)}.parquet"
        if not part.exists():
            ids = ",".join(str(x) for x in dr2_ids[i:i + BATCH])
            q = ("SELECT n.dr2_source_id, n.angular_distance, g.source_id, g.ruwe, g.phot_variable_flag, "
                 "g.non_single_star FROM gaiadr3.dr2_neighbourhood AS n "
                 "JOIN gaiadr3.gaia_source AS g ON g.source_id = n.dr3_source_id "
                 f"WHERE n.dr2_source_id IN ({ids})")
            r = retry(lambda: Gaia.launch_job(q, verbose=False).get_results().to_pandas())
            r.columns = [c.lower() for c in r.columns]
            r.to_parquet(part)
            print(f"  Gaia {min(i + BATCH, len(dr2_ids))}/{len(dr2_ids)}", flush=True)
        frames.append(pd.read_parquet(part))
    df = pd.concat(frames, ignore_index=True)
    df["dr2_source_id"] = df["dr2_source_id"].astype("Int64")
    # a DR2 source can have several DR3 neighbours: keep the closest
    return df.sort_values("angular_distance").drop_duplicates("dr2_source_id")


def clean(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, bytes):
        v = v.decode()
    return v


def main() -> None:
    tics = [int(x) for x in (FOLDER / "hosts.txt").read_text().split()]
    tic = tic_rows(tics)
    gaia = gaia_rows(sorted(int(x) for x in tic["dr2_source_id"].dropna()))
    df = tic.merge(gaia, on="dr2_source_id", how="left")
    stars = {}
    for r in df.itertuples():
        star = {"tic": int(r.tic), "source_id": clean(r.source_id), "tmag": clean(r.Tmag), "teff": clean(r.Teff),
                "rad": clean(r.rad), "mass": clean(r.mass), "distance": clean(r.d), "contratio": clean(r.contratio),
                "ruwe": clean(r.ruwe), "phot_variable_flag": clean(r.phot_variable_flag),
                "non_single_star": clean(r.non_single_star), "ra": clean(r.ra), "dec": clean(r.dec),
                "upcoming_sectors": []}
        if star["source_id"] is not None and not pd.isna(star["source_id"]):
            star["source_id"] = int(star["source_id"])
        else:
            star["source_id"] = None
        if star["mass"] is None and star["rad"]:
            star["mass"], star["mass_estimated"] = star["rad"], True  # as star_info does for dwarfs
        stars[str(star["tic"])] = star
    (FOLDER / "stars.json").write_text(json.dumps(stars), encoding="utf-8")
    have = [s for s in stars.values() if s["rad"] and s["ra"] is not None]
    print(f"{len(stars)}/{len(tics)} hosts in the TIC; {len(have)} with a radius; "
          f"{sum(s['source_id'] is not None for s in stars.values())} matched to Gaia DR3; "
          f"missing: {len(set(tics) - {int(t) for t in stars})}")


if __name__ == "__main__":
    main()
