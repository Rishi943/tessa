"""Known-object catalogs used by the novelty checks.

Each list is downloaded once and cached in data/catalogs/. Pass max_age_days=0
to force a fresh copy (novelty check B does this for every detection).
"""
import time
from pathlib import Path

import pandas as pd
import requests
import truststore

from .paths import CATALOGS

# tess.mit.edu serves an incomplete certificate chain; the OS trust store can complete it.
truststore.inject_into_ssl()

TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
CTOI_URL = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
CONFIRMED_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    "query=select+pl_name,hostname,tic_id,pl_orbper+from+pscomppars&format=csv"
)
TWO_MIN_URL = "https://tess.mit.edu/public/target_lists/2m/all_targets_S{sector:03d}_v1.csv"


def _cached(name: str, fetch, max_age_days: float) -> Path:
    path = CATALOGS / name
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
        return path
    data = fetch()
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def _get(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content


def _sexagesimal(values: pd.Series, scale: float) -> pd.Series:
    """"hh:mm:ss" or "dd:mm:ss" to degrees (scale 15 for hours)."""
    parts = values.astype(str).str.split(":", expand=True).astype(float)
    sign = values.astype(str).str.strip().str.startswith("-").map({True: -1.0, False: 1.0})
    return sign * scale * (parts[0].abs() + parts[1] / 60 + parts[2] / 3600)


def toi(max_age_days: float = 7) -> pd.DataFrame:
    """TESS Objects of Interest: tic, toi, period (d), disposition, ra, dec (deg)."""
    df = pd.read_csv(_cached("toi.csv", lambda: _get(TOI_URL), max_age_days))
    return pd.DataFrame({
        "tic": df["TIC ID"].astype("int64"),
        "toi": df["TOI"],
        "period": pd.to_numeric(df["Period (days)"], errors="coerce"),
        "disposition": df["TFOPWG Disposition"],
        "ra": _sexagesimal(df["RA"], 15.0),
        "dec": _sexagesimal(df["Dec"], 1.0),
    })


def ctoi(max_age_days: float = 7) -> pd.DataFrame:
    """Community TOIs: tic, ctoi, period (d), ra, dec (deg)."""
    df = pd.read_csv(_cached("ctoi.csv", lambda: _get(CTOI_URL), max_age_days))
    return pd.DataFrame({
        "tic": df["TIC ID"].astype("int64"),
        "ctoi": df["CTOI"],
        "period": pd.to_numeric(df["Period (days)"], errors="coerce"),
        "ra": pd.to_numeric(df["RA"], errors="coerce"),
        "dec": pd.to_numeric(df["Dec"], errors="coerce"),
    })


def confirmed(max_age_days: float = 7) -> pd.DataFrame:
    """Confirmed planets with a TIC ID (NASA Exoplanet Archive): tic, planet, period (d)."""
    df = pd.read_csv(_cached("confirmed.csv", lambda: _get(CONFIRMED_URL), max_age_days))
    df = df.dropna(subset=["tic_id"])
    return pd.DataFrame({
        "tic": df["tic_id"].str.replace("TIC ", "", regex=False).astype("int64"),
        "planet": df["pl_name"],
        "period": df["pl_orbper"],
    })


def tess_ebs(max_age_days: float = 365) -> pd.DataFrame:
    """TESS eclipsing binaries (Prsa et al. 2022, VizieR J/ApJS/258/16): tic, period (d)."""

    def fetch() -> bytes:
        from astroquery.vizier import Vizier

        t = Vizier(columns=["TIC", "Per"], row_limit=-1).get_catalogs("J/ApJS/258/16/tess-ebs")[0]
        return t.to_pandas().to_csv(index=False).encode()

    df = pd.read_csv(_cached("tess_ebs.csv", fetch, max_age_days))
    return pd.DataFrame({"tic": df["TIC"].astype("int64"), "period": df["Per"]})


def two_min_targets(last_sector: int, max_age_days: float = 365) -> pd.DataFrame:
    """Every star given 2-minute (or faster) cadence, one row per (tic, sector)."""
    frames = []
    for sector in range(1, last_sector + 1):
        url = TWO_MIN_URL.format(sector=sector)
        try:
            path = _cached(f"2min/S{sector:03d}.csv", lambda: _get(url), max_age_days)
        except requests.HTTPError as e:
            print(f"  2-min list for sector {sector} unavailable ({e.response.status_code}), skipped")
            continue
        df = pd.read_csv(path, comment="#", usecols=["TICID"])
        frames.append(pd.DataFrame({"tic": df["TICID"].astype("int64"), "sector": sector}))
    return pd.concat(frames, ignore_index=True).drop_duplicates()


(CATALOGS / "2min").mkdir(exist_ok=True)
