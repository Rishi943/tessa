"""Export the survey for the site.

sky.json has every star in the pool, column by column, with a status code; summary.json has the
funnel and the counts; stars/TIC.json has, for every star that passed gate 1, its light curve
(binned and folded), the code checks, the agent's dossier, the gate 2 status (with the case-closed line
for rejected stars and the tools' evidence for investigated ones) and, for candidates, the novelty
check and the predictions. planets.json has every confirmed planet and every TESS candidate still
waiting, from the NASA Exoplanet Archive, with Tessa's candidates among them.

Run: python -m pipeline.export [--out data/site]                  the v10 survey: Glimmer's dossiers, gate 2 and TRICERATOPS
     python -m pipeline.export --run v5 [--model M] [--prompt P]   the first run (Muse Spark dossiers), as the site showed it
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import catalogs
from .paths import CACHE, DATA, ROOT, batches
from .search import detrend
from .vet import load_lc, phase_offset

# (name, search results, vetting folder, light-curve shards) for each searched batch; v10 vets in VET_DIR_v10
RUNS = [(name, res, DATA / "vet" / name, lc) for name, (res, lc) in batches().items()]
# TRICERATOPS on v10's passes (tools/fpp_survivors.py); a gate 2 candidate must clear FPP < 0.5 and NFPP < 0.1
FPP_RUNS = DATA / "survey_v10"
FPP_MAX, NFPP_MAX, NFPP_LIKELY = 0.5, 0.1, 0.001
STATUS = {"pool": 0, "held back": 1, "searched": 2, "rejected": 3, "flagged": 4, "awaiting agent": 7,
          "candidate": 5, "known": 6}
TESS = "Transiting Exoplanet Survey Satellite (TESS)"  # the archive's name for it
# Candidates found before the hackathon (the tess-hunt research). Tessa re-finds them blind, which tests the
# pipeline, but they are not its finds: they export as "known" and stay out of the universe and the counts.
KNOWN_BEFORE = {458191673: "found before the hackathon; Tessa re-found it blind"}
BIN_D = 0.5 / 24  # light-curve bins for the page
FOLD_HALF = 3.0  # folded view: this many transit durations each side
FOLD_BINS = 60


def binned(t: np.ndarray, y: np.ndarray, w: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, ...]:
    i = np.digitize(t, edges) - 1
    ok = (i >= 0) & (i < len(edges) - 1)
    sw = np.bincount(i[ok], w[ok], len(edges) - 1)
    swy = np.bincount(i[ok], w[ok] * y[ok], len(edges) - 1)
    swt = np.bincount(i[ok], w[ok] * t[ok], len(edges) - 1)
    has = sw > 0
    return swt[has] / sw[has], swy[has] / sw[has], 1 / np.sqrt(sw[has])


def light_curve(lc: dict, sig: dict) -> dict:
    """The whole light curve in BIN_D bins (per sector, so gaps stay gaps), and the transit folded."""
    series = {"t": [], "y_ppm": [], "sector": []}
    for s in np.unique(lc["sector"]):
        m = lc["sector"] == s
        t = lc["t"][m]
        tb, yb, _ = binned(t, lc["y"][m], lc["w"][m], np.arange(t.min(), t.max() + BIN_D, BIN_D))
        series["t"] += np.round(tb, 4).tolist()
        series["y_ppm"] += np.round(yb * 1e6).astype(int).tolist()
        series["sector"] += [int(s)] * len(tb)
    x = phase_offset(lc["t"], sig["period"], sig["t0"])
    half = FOLD_HALF * sig["dur"]
    near = np.abs(x) < half
    xb, yb, eb = binned(x[near], lc["y"][near], lc["w"][near], np.linspace(-half, half, FOLD_BINS + 1))
    return {"series": series, "fold": {"x_h": np.round(xb * 24, 3).tolist(), "y_ppm": np.round(yb * 1e6, 1).tolist(),
                                       "err_ppm": np.round(eb * 1e6, 1).tolist()}}


def dossier(tic: int, model: str, prompt: str) -> dict | None:
    """The investigation's dossier when the agent investigated this star, else the vetting one. Glimmer's v10
    dossiers (tools/survey_v10.py) list tools by name and carry no per-call cost: it is paid by GPU time."""
    for version in ("inv1", prompt):
        path = DATA / "dossiers" / f"{tic}.{model}.{version}.json"
        if path.exists():
            run = json.loads(path.read_text(encoding="utf-8"))
            tools = [t if isinstance(t, str) else t["tool"] for t in run.get("tools", [])]
            return {**run["dossier"], "model": run["model"], "prompt": run["prompt"], "cost_usd": run.get("cost", 0.0),
                    "tools_called": tools, "invented": run.get("invented", [])}
    return None


def fpp_results() -> dict[int, dict]:
    """TRICERATOPS per star from every fpp_out* run of v10's passes; a later folder wins for a star run twice."""
    out = {}
    for res in sorted(FPP_RUNS.glob("fpp_out*/results.json")):
        for r in json.loads(res.read_text(encoding="utf-8")):
            if r.get("ran"):
                # TRICERATOPS's own class names FPP < 0.015 and NFPP < 0.001 "validated"; validating a planet also
                # needs high-resolution imaging, which no candidate has, so the site never says it
                cls = "likely planet (below TRICERATOPS's validation thresholds, no imaging yet)" if r["class"] == "validated" else r["class"]
                out[int(r["tic"])] = {"FPP": r["FPP"], "NFPP": r["NFPP"], "class": cls}
    return out


def fpp_tier(f: dict) -> str:
    if f["FPP"] >= FPP_MAX or f["NFPP"] >= NFPP_MAX:
        return "fails"
    return "likely planet" if f["NFPP"] < NFPP_LIKELY else "nearby star to rule out"


def dip_row(page: dict) -> dict:
    """One line per gate-1 pass, for the site's list of dips and Ask Tessa's search."""
    c, d = page["checks"], page["dossier"] or {}
    r = lambda v, n=None: None if v is None else round(v, n)  # Gaia has gaps
    return {"tic": page["tic"], "batch": page["batch"], "status": page["status"],
            "period_d": r(c["signal"]["period"], 5), "depth_ppm": r(c["signal"]["depth_ppm"]),
            "snr": r(c["signal"]["snr"], 1), "rp_rearth": r(c["size"]["rp_rearth"], 2),
            "tmag": r(c["star"]["tmag"], 2), "teff": r(c["star"]["teff"]), "dist_pc": r(c["star"]["distance"], 1),
            "ra": r(c["star"]["ra"], 4), "dec": r(c["star"]["dec"], 4),
            "line": page["case_closed"] or d.get("public_summary") or "; ".join(page["reasons"]),
            "reasons": page["reasons"], "dossier": bool(d), "tools": len(d.get("tools_called", []))}


def universe(out: Path, ours: list[dict]) -> dict:
    """planets.json: every confirmed planet, every TESS candidate still waiting, and Tessa's candidates, column by
    column. kind is 0 confirmed, 1 TESS candidate, 2 Tessa's. Rows without a distance can't be placed and are only
    counted."""
    p, t = catalogs.planets(), catalogs.open_tois()
    # orbits for the system view: a (AU), eccentricity, inclination and periastron angle (degrees), and a mid-transit
    # (tc) or periastron (tp) time in BJD to place each planet where it is today. TESS candidates transit, so they are
    # edge-on; their star's mass comes from its gravity and radius (M = g R^2, in solar units).
    rows = pd.concat([
        pd.DataFrame({"kind": 0, "name": p["pl_name"], "host": p["hostname"], "ra": p["ra"], "dec": p["dec"],
                      "dist": p["sy_dist"], "method": p["discoverymethod"], "year": p["disc_year"],
                      "facility": p["disc_facility"], "period": p["pl_orbper"], "radius": p["pl_rade"],
                      "mass": p["pl_bmasse"], "teq": p["pl_eqt"], "teff": p["st_teff"], "a": p["pl_orbsmax"],
                      "ecc": p["pl_orbeccen"], "incl": p["pl_orbincl"], "omega": p["pl_orblper"], "tc": p["pl_tranmid"],
                      "tp": p["pl_orbtper"], "srad": p["st_rad"], "smass": p["st_mass"]}),
        pd.DataFrame({"kind": 1, "name": "TOI-" + t["toi"].map("{:.2f}".format), "host": "TIC " + t["tid"].astype(str),
                      "ra": t["ra"], "dec": t["dec"], "dist": t["st_dist"], "method": "Transit",
                      "year": t["toi_created"].str[:4].astype(int), "facility": TESS, "period": t["pl_orbper"],
                      "radius": t["pl_rade"], "mass": np.nan, "teq": t["pl_eqt"], "teff": t["st_teff"], "a": np.nan,
                      "ecc": np.nan, "incl": 90.0, "omega": np.nan, "tc": t["pl_tranmid"], "tp": np.nan,
                      "srad": t["st_rad"], "smass": 10 ** (t["st_logg"] - 4.438) * t["st_rad"] ** 2}),
        pd.DataFrame([{"kind": 2, "name": f"TIC {o['tic']}", "host": f"TIC {o['tic']}", "ra": o["checks"]["star"]["ra"],
                       "dec": o["checks"]["star"]["dec"], "dist": o["checks"]["star"]["distance"], "method": "Transit",
                       "year": 2026, "facility": TESS, "period": o["ephem"]["period"],
                       "radius": o["checks"]["size"]["rp_rearth"], "mass": np.nan, "teq": np.nan,
                       "teff": o["checks"]["star"]["teff"], "a": np.nan, "ecc": np.nan, "incl": 90.0, "omega": np.nan,
                       "tc": o["ephem"]["t0"] + 2457000, "tp": np.nan,  # BTJD to BJD
                       "srad": o["checks"]["star"]["rad"], "smass": o["checks"]["star"]["mass"]} for o in ours]),
    ], ignore_index=True)
    placed = rows.dropna(subset=["ra", "dec", "dist"])
    methods = rows.loc[rows["kind"] == 0, "method"].value_counts().index.tolist()
    facilities = placed["facility"].value_counts().index.tolist()
    col = lambda name, nd: [None if pd.isna(v) else round(float(v), nd) for v in placed[name]]
    data = {"methods": methods, "facilities": facilities,
            "counts": {"confirmed": int((rows["kind"] == 0).sum()), "waiting": int((rows["kind"] == 1).sum()),
                       "tessa": len(ours), "unplaced": {"confirmed": int((p["sy_dist"].isna()).sum()),
                                                          "waiting": int((t["st_dist"].isna()).sum())},
                       "by_method": {m: {"planets": int(((rows["kind"] == 0) & (rows["method"] == m)).sum()),
                                         "first": int(rows.loc[(rows["kind"] == 0) & (rows["method"] == m), "year"].min())}
                                     for m in methods}},
            "kind": placed["kind"].tolist(), "name": placed["name"].tolist(), "host": placed["host"].tolist(),
            "ra": col("ra", 4), "dec": col("dec", 4), "dist": col("dist", 2),
            "method": placed["method"].map(methods.index).tolist(), "year": placed["year"].astype(int).tolist(),
            "facility": placed["facility"].map(facilities.index).tolist(), "period": col("period", 5),
            "radius": col("radius", 2), "mass": col("mass", 2), "teq": col("teq", 0), "teff": col("teff", 0),
            "a": col("a", 5), "ecc": col("ecc", 3), "incl": col("incl", 2), "omega": col("omega", 1), "tc": col("tc", 5),
            "tp": col("tp", 5), "srad": col("srad", 3), "smass": col("smass", 3)}
    (out / "planets.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data["counts"]


def export(out: Path, model: str, prompt: str, v10: bool = True) -> dict:
    (out / "stars").mkdir(parents=True, exist_ok=True)
    fpp = fpp_results() if v10 else {}
    pool = pd.read_parquet(DATA / "targets.parquet", columns=["tic", "ra", "dec", "parallax", "tmag", "teff", "rad",
                                                              "two_min_sectors"])
    status = pd.Series(STATUS["pool"], index=pool["tic"])
    held = {int(t) for t in (ROOT / "survey" / "holdout.txt").read_text(encoding="utf-8").split()}
    status[status.index.isin(held)] = STATUS["held back"]

    counts = {}
    dips, ours = [], []
    costs = {"vetting": 0.0, "investigation": 0.0, "case_closed": 0.0, "observing": 0.0, "audit": 0.0}
    invented = {"dossiers": 0, "with_invented_numbers": 0}  # numbers a model wrote that were not in its input
    glimmer = {}  # v10: Glimmer's verdicts on the signals that reached it (gate2 = it called a planet)
    for name, results, vet_dir, lc_dir in RUNS:
        vet_dir = vet_dir.with_name(f"{vet_dir.name}_v10") if v10 else vet_dir
        if not results.exists():
            continue
        for step, csv_name in (("vetting", f"dossiers.{model}.{prompt}.csv"), ("investigation", f"investigations.{model}.inv1.csv")):
            if (vet_dir / csv_name).exists():
                runs = pd.read_csv(vet_dir / csv_name)
                costs[step] += float(runs["cost"].fillna(0).sum())
                invented["dossiers"] += len(runs)
                invented["with_invented_numbers"] += int((runs["invented"].fillna(0) > 0).sum())
        if (vet_dir / "caseclosed.meta.json").exists():
            costs["case_closed"] += json.loads((vet_dir / "caseclosed.meta.json").read_text(encoding="utf-8"))["cost_usd"]
        res = pd.read_csv(results)
        status[status.index.isin(res.loc[res["n_points"].fillna(0) > 0, "tic"])] = STATUS["searched"]
        final = {}
        closed = json.loads((vet_dir / "caseclosed.json").read_text(encoding="utf-8"))             if (vet_dir / "caseclosed.json").exists() else {}
        if (vet_dir / "status.csv").exists():
            with (vet_dir / "status.csv").open(newline="", encoding="utf-8") as f:
                final = {int(r["tic"]): r for r in csv.DictReader(f)}
        for _, r in res[res["gate1"] == "pass"].iterrows():
            tic = int(r["tic"])
            st = final.get(tic, {"status": "awaiting agent", "reasons": ""})
            if v10 and st["status"] == "candidate" and tic in fpp and fpp_tier(fpp[tic]) == "fails":
                f = fpp[tic]
                st = {**st, "status": "flagged",
                      "reasons": f"TRICERATOPS: FPP {f['FPP']:.3f}, NFPP {f['NFPP']:.1e}, above the bar (FPP < {FPP_MAX}, NFPP < {NFPP_MAX})"}
            if tic in KNOWN_BEFORE and st["status"] == "candidate":
                st = {**st, "status": "known", "reasons": KNOWN_BEFORE[tic]}
            status[tic] = STATUS[st["status"]]
            counts[st["status"]] = counts.get(st["status"], 0) + 1
            checks = json.loads((vet_dir / f"{tic}.json").read_text(encoding="utf-8"))
            if v10 and st["status"] == "awaiting agent" and not st["reasons"]:  # set aside: say why
                from .gate2 import evidence, outside_planet_range
                why = outside_planet_range(checks, evidence(tic), "v10")
                st = {**st, "reasons": "; ".join(why) if why else
                      "Glimmer answered, but its reply could not be read, so it made no call"}
            sig = {k: checks["signal"][k] for k in ("period", "t0", "dur")}
            cand = vet_dir / "candidates" / f"{tic}.json"
            c = json.loads(cand.read_text(encoding="utf-8")) if cand.exists() else {}
            eph = (c.get("when") or {}).get("ephemeris") or {}
            if eph.get("ok"):  # the refitted ephemeris centres the fold better than the search grid
                sig = {**sig, "period": eph["period"], "t0": eph["t0_btjd"]}
            raw = load_lc(tic, lc_dir)
            page = {"tic": tic, "batch": name, "status": st["status"],
                    "reasons": [x for x in st["reasons"].split("; ") if x],
                    "gate1": {"dip_snr": r["dip_snr"], "flipped_snr": r["bump_snr"], "transits": int(r["dip_transits"])},
                    "case_closed": closed.get(str(tic)), "checks": checks, "dossier": dossier(tic, model, prompt),
                    "ephem": {"period": sig["period"], "t0": sig["t0"]},
                    **light_curve(detrend(raw["time"], raw["flux"], raw["sector"]), sig)}
            if tic in fpp:
                page["fpp"] = {**fpp[tic], "tier": fpp_tier(fpp[tic])}
            if v10 and page["dossier"]:  # Glimmer's dossiers carry their own check (finetune/pilot.py invented)
                verdict = page["dossier"].get("verdict")
                glimmer[verdict] = glimmer.get(verdict, 0) + 1
                invented["dossiers"] += 1
                invented["with_invented_numbers"] += bool(page["dossier"].get("invented"))
            ev = DATA / "evidence" / f"{tic}.json"
            if ev.exists():
                page["evidence"] = json.loads(ev.read_text(encoding="utf-8"))
            if c:
                page["novelty"], page["when"], page["observing"] = c.get("novelty"), c.get("when"), c.get("observing")
                costs["observing"] += (c.get("observing") or {}).get("cost_usd", 0)
                # second opinion, shortlist only; Wonder audited the first run's candidates (prompt v5)
                audit = next((f for v in (prompt, "v5") if (f := DATA / "dossiers" / f"{tic}.muse-wonder-1.0.{v}.json").exists()),
                             DATA / "dossiers" / "none")
                if audit.exists():
                    run = json.loads(audit.read_text(encoding="utf-8"))
                    page["audit"] = {**run["dossier"], "model": run["model"], "cost_usd": run["cost"]}
                    costs["audit"] += run["cost"]
            (out / "stars" / f"{tic}.json").write_text(json.dumps(page, separators=(",", ":"), default=float),
                                                         encoding="utf-8")
            dips.append(dip_row(page))
            if page["status"] == "candidate":
                ours.append(page)
        gate = vet_dir.parent.parent / f"{name}_gate1.json"
        counts[f"{name}_searched"] = int((res["n_points"].fillna(0) > 0).sum())  # a star with no light curve was not searched
        if gate.exists():
            counts[f"{name}_gate1"] = json.loads(gate.read_text(encoding="utf-8"))

    sky = pool.assign(status=pool["tic"].map(status).astype(int))
    (out / "sky.json").write_text(json.dumps({
        "tic": sky["tic"].tolist(), "ra": sky["ra"].round(4).tolist(), "dec": sky["dec"].round(4).tolist(),
        "dist": (1000 / sky["parallax"]).round(2).tolist(), "tmag": sky["tmag"].round(2).tolist(),
        "teff": sky["teff"].round(0).astype(int).tolist(), "status": sky["status"].tolist(), "codes": STATUS},
        separators=(",", ":")), encoding="utf-8")
    (out / "dips.json").write_text(json.dumps(dips, separators=(",", ":"), default=float), encoding="utf-8")
    # Any star by TIC, in 100 small files, so Ask Tessa can say what happened to a star without a case file
    (out / "pool").mkdir(exist_ok=True)
    names = {v: k for k, v in STATUS.items()}
    for shard, part in sky.groupby(sky["tic"] % 100):
        rows = {int(r.tic): {"status": names[r.status], "dist_pc": round(1000 / r.parallax, 1), "tmag": round(r.tmag, 2),
                             "teff": None if pd.isna(r.teff) else int(r.teff), "ra": round(r.ra, 4), "dec": round(r.dec, 4)}
                for r in part.itertuples()}
        (out / "pool" / f"{shard}.json").write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    summary = {"pool": len(pool), "funnel": json.loads((DATA / "targets_funnel.json").read_text(encoding="utf-8")),
               "held_back": len(held), "by_status": {k: int((sky["status"] == v).sum()) for k, v in STATUS.items()},
               "costs_usd": {k: round(v, 4) for k, v in costs.items()}, "invented": invented,
               "glimmer": {"reached": sum(glimmer.values()), "called_planet": glimmer.get("gate2", 0),
                           "flagged": glimmer.get("flag", 0), "rejected": glimmer.get("reject", 0)} if glimmer else None,
               "dossiers": sum(d["dossier"] for d in dips), "investigated": sum(d["tools"] > 0 for d in dips),
               "universe": universe(out, ours), **counts,
               # every batch together, for the tour's numbers
               "searched_total": sum(v for k, v in counts.items() if k.endswith("_searched")),
               "gate1_total": {k: sum(v[k] for n, v in counts.items() if n.endswith("_gate1")) for k in ("pass", "flipped_pass")}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DATA / "site"))
    ap.add_argument("--run", default="v10", choices=["v10", "v5"], help="v10: Glimmer's survey; v5: the first run")
    ap.add_argument("--model", help="dossier model (default: glimmer-vetter-v10 for v10, muse-spark-1.3-contributor for v5)")
    ap.add_argument("--prompt", help="dossier prompt (default: the run's)")
    args = ap.parse_args()
    v10 = args.run == "v10"
    model = args.model or ("glimmer-vetter-v10" if v10 else "muse-spark-1.3-contributor")
    prompt = args.prompt or args.run
    print(json.dumps(export(Path(args.out), model, prompt, v10), indent=1, default=float))


if __name__ == "__main__":
    main()
