"""Extra clues for System 1, for every labelled TOI: a Kaggle CPU job next to finetune/inputs.py.

1. Centroid (difference imaging) for TOIs whose code rules rejected them before it ran.
2. A limb-darkened transit fit (numpy only: exact uniform-disk overlap, limb darkening at the planet's
   position, exposure smearing): radius ratio, impact parameter, a/R*, and the star density it implies
   against the TIC's (an eclipsing binary or a blend rarely matches).
3. Gaia DR3: radial-velocity scatter (rv_chisq_pvalue, rv_renormalised_gof, rv_amplitude_robust: a
   spectroscopic binary shows up here), image doubling (ipd_frac_multi_peak, ipd_gof_harmonic_amplitude),
   astrometric excess noise, FLAME / GSP-Phot radius; and whether the target or a nearby suspect is in
   Gaia's eclipsing-binary list at this period or a harmonic.
4. TOIs vetted without a TIC radius: Gaia's radius instead, then the full vet as in inputs.py.

Run (Kaggle, with tessa-ft-fetch and tessa-ft-vet-0..2 mounted):
  python ft_clues.py --labels labels.json --stars stars.json --out /kaggle/working --part 0 --parts 3
Writes clues/TOI.json, vet/TOI.json (newly vetted TOIs only) and gaia_PART.json.
"""
import argparse
import json
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from pipeline.fit import eb_match, gaia_ebs, gaia_extra, model, overlap, transit_fit  # noqa: F401  (moved there)

T0 = time.time()


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f} min  {msg}", flush=True)


# --- per TOI -----------------------------------------------------------------------------------

def one(job: tuple) -> dict:
    from pipeline.search import detrend
    from pipeline.vet import finite
    vet_path, label, star, shard = job
    t_start = time.time()
    out = {"toi": label["toi"]}
    try:
        if vet_path is None:  # no TIC radius before: vet it now with Gaia's
            from finetune.inputs import one as vet_one
            row = vet_one((label, star, shard))
            out["vetted_now"] = row.get("status")
            vet_path = Path(os.environ["FT_OUT"], "vet", f"{label['toi']}.json")
            if not vet_path.exists():
                return out
        d = json.loads(Path(vet_path).read_text(encoding="utf-8"))
        checks = d["checks"]
        with np.load(shard) as z:
            raw = {k: z[f"{label['tic']}_{k}"] for k in ("time", "flux", "sector", "source")}
        sig = {**checks["signal"]}
        sig["dur"] = sig.get("dur") or sig["dur_h"] / 24
        if not checks.get("centroid"):
            from pipeline.centroid import where
            out["centroid"] = finite(where(label["tic"], sig, star=checks["star"],
                                           sectors=sorted({int(s) for s in raw["sector"]})))
        lc = detrend(raw["time"], raw["flux"], raw["sector"])
        out["fit"] = finite(transit_fit(lc, sig, checks["star"]))
    except Exception as e:  # one bad star must not stop the job
        out["error"] = f"{e.__class__.__name__}: {e}"[:300]
    out["seconds"] = round(time.time() - t_start, 1)
    Path(os.environ["FT_OUT"], "clues", f"{label['toi']}.json").write_text(json.dumps(out, default=float), encoding="utf-8")
    return out


def main() -> None:
    from finetune.inputs import shards_by_tic
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", required=True)
    ap.add_argument("--stars", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--inputs", default=os.environ.get("KAGGLE_INPUTS", "/kaggle/input"))
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-minutes", type=float, default=600)
    args = ap.parse_args()

    out = Path(args.out)
    for sub in ("clues", "vet"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    os.environ["FT_OUT"] = str(out)
    # labels and stars of TOIs vetted before come from their vet files (mounted); the bundled files hold
    # only the rest, which keeps the upload small
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    stars = json.loads(Path(args.stars).read_text(encoding="utf-8"))
    root = Path(args.inputs)
    shards = shards_by_tic(root)
    # earlier vet outputs, by TOI ("1008.01.json"); the mount may or may not keep their vet/ folder
    vets = {p.stem: p for p in root.rglob("*.json") if re.fullmatch(r"\d+\.\d\d", p.stem) and "tessa-ft-vet" in str(p)}
    known = {r["toi"] for r in labels}
    for path in vets.values():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d["toi"] not in known:
            labels.append(d["label"])
        stars.setdefault(str(d["tic"]), d["checks"]["star"])
    log(f"{len(vets)} earlier vet files found, {len(labels)} labels")
    hosts = set(sorted({r["tic"] for r in labels})[args.part::args.parts])
    mine = sorted((r for r in labels if r["tic"] in hosts and r["tic"] in shards), key=lambda r: r["toi"])[:args.limit]
    log(f"part {args.part}/{args.parts}: {len(mine)} TOIs with light curves, {sum(r['toi'] in vets for r in mine)} vetted before")

    # Gaia extras for every host in this part, and EB periods for each target and its nearby suspects
    ids = sorted({int(stars[str(r["tic"])]["source_id"]) for r in mine if stars.get(str(r["tic"]), {}).get("source_id")})
    extra = gaia_extra(ids)
    suspects = set(ids)
    for r in mine:
        if r["toi"] in vets:
            c = json.loads(vets[r["toi"]].read_text(encoding="utf-8"))["checks"]
            suspects |= {int(n["source_id"]) for n in c["neighbours"].get("closest_that_could") or [] if n.get("source_id")}
            suspects |= {int(n["source_id"]) for n in (c.get("centroid") or {}).get("suspects") or [] if n.get("source_id")}
    ebs = gaia_ebs(sorted(suspects))
    (out / f"gaia_{args.part}.json").write_text(json.dumps({"extra": extra, "eb": ebs}), encoding="utf-8")

    jobs = []
    for r in mine:
        star = dict(stars.get(str(r["tic"])) or {})
        if r["toi"] in vets:
            jobs.append((str(vets[r["toi"]]), r, star, shards[r["tic"]]))
            continue
        g = extra.get(str(star.get("source_id")), {})
        rad = g.get("radius_flame") or g.get("radius_gspphot")
        if not rad:
            continue
        star.update(rad=rad, rad_source="gaia_flame" if g.get("radius_flame") else "gaia_gspphot")
        star["mass"] = star.get("mass") or g.get("mass_flame") or rad ** 1.25  # main-sequence guess, flagged
        star["mass_estimated"] = not g.get("mass_flame")
        if star.get("ra") is None:
            star["ra"], star["dec"] = g.get("ra"), g.get("dec")
        if star.get("ra") is not None:
            jobs.append((None, r, star, shards[r["tic"]]))
    log(f"{len(jobs)} TOIs to work on ({sum(j[0] is None for j in jobs)} newly vettable with Gaia's radius)")

    done = {"centroid": 0, "fit ok": 0, "errors": 0, "newly vetted": 0}
    with ProcessPoolExecutor(args.workers, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(one, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            done["centroid"] += "centroid" in r
            done["fit ok"] += (r.get("fit") or {}).get("status") == "ok"
            done["errors"] += "error" in r
            done["newly vetted"] += r.get("vetted_now") == "ok"
            if n % 50 == 0:
                log(f"{n}/{len(jobs)} {done}")
            if (time.time() - T0) / 60 > args.max_minutes:
                log("time limit reached; cancelling the rest")
                for g in futures:
                    g.cancel()
                break
    log(f"done: {done}")


if __name__ == "__main__":
    main()
