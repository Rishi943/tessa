"""Training labels for the Glimmer vetter: TOIs whose fate follow-up already settled.

TFOPWG dispositions: CP and KP are planets (dossier target gate2), FP and FA are not (reject),
APC is ambiguous (flag); PC has no outcome yet and is left out. Only orbits the survey's search
can find (0.5-15 days) are kept. Splits are by host star, so no star is in two of them: TOIs
alerted from 2024 on and the stars of our labelled set always go to test, the rest is split
70/10/20 stratified by label.

Run: python -m finetune.labels data/finetune/toi_exofop_2026-10-28.csv
Writes data/finetune/labels.csv and data/finetune/hosts.txt (one TIC per line, for fetch.py).
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.paths import DATA

TARGET = {"CP": "gate2", "KP": "gate2", "FP": "reject", "FA": "reject", "APC": "flag"}
P_RANGE = (0.5, 15.0)  # the search's period range
RECENT = "2024-01-01"  # alerted on or after this: test only (least likely to be in pretraining)
SPLIT = {"train": 0.7, "val": 0.1, "test": 0.2}
SEED = 20261028
BTJD = 2457000.0

# Why a TOI failed, from its ExoFOP comment; first match wins. "spectra" are false positives
# that only spectroscopy shows (TESS photometry alone can't), so they bound what vetting can do.
REASONS = [
    ("blend", r"\bNEB\b|nearby EB|centroid|offset|contaminat|blend|on (?:a |an )?(?:nearby|neighbo)"),
    ("spectra", r"\bSB1\b|\bSB2\b|EBLM|\bRV\b|radial velocit|spectr|CORALIE|TRES|CHIRON|FEROS|HARPS|composite"),
    ("binary", r"\bEB\b|eclips|V-shaped|v-shape|secondary|odd.even|odd/even|ellipsoidal|binary"),
    ("giant", r"giant|evolved|subgiant|large star|Rs ?>"),
    ("false alarm", r"no detection|not detected|systematic|scattered light|momentum dump|low SNR|variabil|"
                    r"not recovered|spurious|instrumental"),
]


def reason(comment: str) -> str:
    if not isinstance(comment, str):
        return "unknown"
    for name, pattern in REASONS:
        if re.search(pattern, comment, re.IGNORECASE):
            return name
    return "unknown"


def split_hosts(hosts: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """hosts: one row per star with its label; returns train/val/test per star."""
    out = pd.Series("test", index=hosts.index)
    free = hosts[~hosts["forced_test"]]
    for _, group in free.groupby("label"):
        order = rng.permutation(group.index.to_numpy())
        n_train = int(round(SPLIT["train"] * len(order)))
        n_val = int(round(SPLIT["val"] * len(order)))
        out[order[:n_train]] = "train"
        out[order[n_train:n_train + n_val]] = "val"
    return out


def build(exofop: Path) -> pd.DataFrame:
    df = pd.read_csv(exofop, comment="#", low_memory=False)
    df = df[df["TFOPWG Disposition"].isin(TARGET)].copy()
    out = pd.DataFrame({
        "toi": df["TOI"].astype(str), "tic": df["TIC ID"].astype("int64"), "disp": df["TFOPWG Disposition"],
        "period": df["Period (days)"], "period_err": df["Period (days) err"],
        "t0_btjd": df["Epoch (BJD)"] - BTJD, "t0_err": df["Epoch (BJD) err"],
        "dur_h": df["Duration (hours)"], "depth_ppm": df["Depth (ppm)"], "snr_toi": df["Planet SNR"],
        "tmag": df["TESS Mag"], "teff": df["Stellar Eff Temp (K)"], "rad": df["Stellar Radius (R_Sun)"],
        "source": df["Source"].astype(str).str.split("-").str[0], "sectors": df["Sectors"],
        "alerted": pd.to_datetime(df["Date TOI Alerted (UTC)"], errors="coerce").dt.date.astype(str),
        "comment": df["Comments"],
    })
    out["target"] = out["disp"].map(TARGET)
    out["reason"] = np.where(out["target"] == "gate2", "", out["comment"].map(reason))
    n0 = len(out)
    out = out[out["period"].between(*P_RANGE) & (out["dur_h"] > 0) & out["t0_btjd"].notna()]
    print(f"{n0} labelled TOIs, {len(out)} with a 0.5-15 d orbit and an ephemeris")

    labelled = {int(x) for x in (DATA / "labelled.txt").read_text().split()}
    hosts = out.groupby("tic").agg(label=("target", lambda s: s.mode().iloc[0]), alerted=("alerted", "max"))
    hosts["forced_test"] = (hosts["alerted"] >= RECENT) | hosts.index.isin(labelled)
    hosts["split"] = split_hosts(hosts, np.random.default_rng(SEED))
    out["split"] = out["tic"].map(hosts["split"])
    out["recent"] = out["alerted"] >= RECENT
    return out.sort_values(["split", "tic", "toi"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("exofop", help="ExoFOP TOI table (CSV snapshot)")
    args = ap.parse_args()
    out = build(Path(args.exofop))
    folder = DATA / "finetune"
    folder.mkdir(exist_ok=True)
    out.to_csv(folder / "labels.csv", index=False)
    (folder / "hosts.txt").write_text("\n".join(str(t) for t in out["tic"].unique()) + "\n")
    # the Kaggle job's copy (code bundles take JSON, not CSV); NaN becomes null
    records = json.loads(out.to_json(orient="records"))
    (folder / "labels.json").write_text(json.dumps(records), encoding="utf-8")
    print(pd.crosstab(out["split"], out["target"], margins=True).to_string())
    print(pd.crosstab(out["reason"], out["disp"]).to_string())
    print(f"{out['tic'].nunique()} hosts; recent (test only): {int(out['recent'].sum())} TOIs; "
          f"K/M hosts: {int((out['teff'] < 5300).sum())}")


if __name__ == "__main__":
    main()
