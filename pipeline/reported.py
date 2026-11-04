"""Signals other searches have already reported: published candidate lists beyond ExoFOP.

A match here doesn't make a signal "known" in gate 2: these lists are new and scattered, and a search can
find a signal independently of one it never saw. Novelty records the match ("also reported by"), so the
site and preprints can credit it.

Lists, reduced to one row per star and period (tic, period, list):
  - eumemic/exoplanets (GitHub): the counted candidates of its full-sky and earlier searches, any tier
  - RAVEN (Zenodo 19661443): the TESS-SPOC search's vetted, validated and NSFP < 0.09 tables
  - T16 (Roth et al. 2026, Harvard Dataverse): its candidates, and single transits (no period)
  - ExoMiner++ (Zenodo 15466293) and ExoMiner++ 2.0 (Zenodo 17707413): SPOC TCEs it scores as planets (>= 0.5)
  - Kunimoto et al. 2025 (AJ 170, 280; not on VizieR): LEO-Vetter's 172 M-dwarf candidates, the journal's Table 2
  - Single-star Zenodo records (RECORDS below), found by searching Zenodo for each candidate's TIC on 2026-11-04

Run (Kaggle CPU, as the files are about 220 MB): python -m pipeline.reported --build OUT_DIR
     writes reported.csv (a few MB) and a log of each list's columns; copy it to data/catalogs/reported.csv
"""
import argparse
import io
import json
import re
import sys

import pandas as pd

from .paths import CATALOGS

TABLE = CATALOGS / "reported.csv"
# Zenodo records about one or a few stars, not in any list above: (tic, period or None, credit). A period of None
# credits the star whatever its period (the record gives none).
RECORDS = [
    (9066956, 0.71345, "PHANTOM (Benoit 2026, doi 10.5281/zenodo.23584959)"),
    (116046773, 5.650353, "Reckling 2026 (doi 10.5281/zenodo.23664505)"),
    (50099421, None, "Ibrahim 2026 (doi 10.5281/zenodo.23569126)"),
]
EUMEMIC = "https://raw.githubusercontent.com/eumemic/exoplanets/main/results_public/{}"
RAVEN = "https://zenodo.org/api/records/19661443/files/{}/content"
T16 = ("https://dataverse.harvard.edu/api/access/datafile/13605255?format=original",
       "https://dataverse.harvard.edu/api/access/datafile/13605256?format=original")
EXOMINER = [("https://zenodo.org/api/records/15466293/files/exominerplusplus_catalog_unk_tces_s1-s67_"
             "tess-spoc-2min_complete_1-16-2025_1014.csv/content", "ExoMiner++"),
            ("https://zenodo.org/api/records/17707413/files/vetting_catalog_2-min.csv/content", "ExoMiner++ 2.0 (2-min)"),
            ("https://zenodo.org/api/records/17707413/files/vetting_catalog_FFI.csv/content", "ExoMiner++ 2.0 (FFI)")]
KUNIMOTO = "https://content.cld.iop.org/journals/1538-3881/170/5/280/revision1/ajae070at2_mrt.txt"
EXOMINER_PLANET = 0.5


def get(url: str) -> bytes:
    import requests
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt == 2:
                raise


def column(df: pd.DataFrame, *words: str) -> str:
    """The first column whose lower-cased name contains every word."""
    return next(c for c in df.columns if all(w in c.lower() for w in words))


def rows(df: pd.DataFrame, tic: str, period: str | None, label: str) -> pd.DataFrame:
    out = pd.DataFrame({"tic": pd.to_numeric(df[tic], errors="coerce"),
                        "period": pd.to_numeric(df[period], errors="coerce") if period else float("nan"), "list": label})
    return out.dropna(subset=["tic"]).astype({"tic": "int64"})


def build(out_dir: str) -> None:
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables, log = [], {}

    def add(label, fetch):
        try:
            df = fetch()
            tables.append(df)
            log[label] = f"{len(df)} rows"
        except Exception as e:  # one list's failure must not lose the others
            log[label] = f"{e.__class__.__name__}: {e}"[:300]
        print(label, log[label], flush=True)

    def eumemic():
        parts = []
        for name, label in (("candidates_allsky.csv", "eumemic full-sky"), ("tiers_earlier_candidates.csv", "eumemic earlier")):
            d = pd.read_csv(io.BytesIO(get(EUMEMIC.format(name))))
            log[f"{label} columns"] = list(d.columns)
            d = d[d["tier"].astype(str).str.fullmatch(r"[MSNAB]|L1")]  # the counted tiers (A, B before 27 October)
            parts.append(rows(d, "TIC", "P_d", f"{label} ({len(d)} counted)"))
        return pd.concat(parts)

    def raven():
        parts = []
        for name in ("vet_table.csv", "val_table.csv", "val_rp8_table.csv", "nsfp09_table.csv"):
            d = pd.read_csv(io.BytesIO(get(RAVEN.format(name))))
            log[f"RAVEN {name} columns"] = list(d.columns)
            parts.append(rows(d, column(d, "tic"), column(d, "per"), f"RAVEN {name.split('_table')[0]}"))
        return pd.concat(parts)

    def t16():
        parts = []
        for url in T16:
            raw = get(url)
            d = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
            log[f"T16 {url[-20:]} columns"] = list(d.columns)
            per = next((c for c in d.columns if c.lower().startswith("p_") or c.lower() in ("period", "per")), None)
            parts.append(rows(d, column(d, "tic"), per, "T16" if per else "T16 single transit"))
        return pd.concat(parts)

    def exominer(url, label):
        def fetch():
            d = pd.read_csv(io.BytesIO(get(url)), low_memory=False, comment="#")  # 2.0 files open with comment lines
            log[f"{label} columns"] = list(d.columns)
            score = next(c for c in d.columns if "score" in c.lower())
            d = d[pd.to_numeric(d[score], errors="coerce") >= EXOMINER_PLANET]
            tic = next(c for c in d.columns if c.lower() in ("target_id", "tic id", "tic_id", "ticid", "tic"))
            return rows(d, tic, column(d, "period"), label)
        return fetch

    def kunimoto():
        # the journal's machine-readable Table 2 (CDS format): a header, then rows of TIC, TOI, CTOI, Per, ...
        r = [line.split() for line in get(KUNIMOTO).decode("utf-8", "replace").splitlines() if re.match(r"\s*\d{5,9}\s+\d+\.\d\d\s", line)]  # TICs are right-aligned
        d = pd.DataFrame({"TIC": [x[0] for x in r], "Per": [x[3] for x in r]})
        log["Kunimoto columns"] = "Table 2: TIC (col 1), Per (col 4)"
        return rows(d, "TIC", "Per", "Kunimoto et al. 2025 (LEO-Vetter M dwarfs)")

    add("eumemic", eumemic)
    add("RAVEN", raven)
    add("T16", t16)
    for url, label in EXOMINER:
        add(label, exominer(url, label))
    add("Kunimoto 2025", kunimoto)
    table = pd.concat(tables).assign(period=lambda d: d["period"].round(6)).drop_duplicates()
    table.to_csv(out / "reported.csv", index=False)
    (out / "reported_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"{len(table)} rows from {table['list'].nunique()} lists", flush=True)


def reported(tic: int, sig: dict) -> list[dict]:
    """The listed signals on this star at this period or a simple harmonic (or with no period)."""
    from .vet import harmonic
    if not TABLE.exists():
        return []
    df = pd.read_csv(TABLE)
    out = []
    for r in df[df["tic"] == tic].itertuples():
        h = harmonic(r.period, sig["period"]) if pd.notna(r.period) else None
        if h is not None or pd.isna(r.period):
            out.append({"list": r.list, "period": None if pd.isna(r.period) else float(r.period), "harmonic": h})
    for t, period, credit in RECORDS:
        if t == tic:
            h = harmonic(period, sig["period"]) if period else None
            if h is not None or period is None:
                out.append({"list": credit, "period": period, "harmonic": h})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", metavar="OUT_DIR", help="fetch every list and write reported.csv there")
    ap.add_argument("tic", type=int, nargs="?")
    ap.add_argument("--period", type=float)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if a.build:
        build(a.build)
    else:
        print(json.dumps(reported(a.tic, {"period": a.period}), indent=1))


if __name__ == "__main__":
    main()
