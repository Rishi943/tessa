"""XGBoost in Glimmer's place, through the whole chain: every survey signal XGBoost alone would pass goes through
gate 2 (novelty, testability, seasons) and TRICERATOPS exactly as Glimmer's passes did, and the candidates are compared.

  python tools/xgb_chain.py OUT_DIR [--jobs 4]
      1. finds the cuts: the v9 cut, the cut matched on val to Glimmer's planet recall, and the highest cut that keeps
         all of Glimmer's candidates;
      2. runs gate 2 on every signal XGBoost passes at the lowest of them that Glimmer did not pass (cached in
         OUT_DIR/gate2.json, so a rerun only does what is missing), and folds each new gate 2 candidate for TRICERATOPS;
      3. prints, for each cut: XGBoost's passes, its gate 2 candidates, and its candidates after TRICERATOPS.
  Then, in the TRICERATOPS environment: python tools/fpp_survivors.py --folded OUT_DIR --out OUT_DIR/fpp
  and run this again to read those results.

Signals Glimmer passed keep their own gate 2 status and TRICERATOPS result; the hard rows apply to XGBoost as to Glimmer
(gate2.outside_planet_range), and TIC 458191673, found before the hackathon, is not counted, as on the site.
"""
import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

KNOWN_BEFORE = 458191673


def tri_ok(r: dict | None) -> bool | None:
    if not r or not r.get("ran"):
        return None
    return r["FPP"] < 0.5 and r["NFPP"] < 0.1


def as_pass(job: tuple[int, str]) -> tuple[int, str, list[str]]:
    """Gate 2 on a signal as if it had been passed (XGBoost in Glimmer's place)."""
    from pipeline import gate2
    from pipeline.paths import DATA, batches
    from pipeline.vet import signal_from_results
    tic, batch = job
    res, lc = batches()[batch]
    c = json.loads((DATA / "vet" / f"{batch}_v10" / f"{tic}.json").read_text(encoding="utf-8"))
    d = {"prompt": "v10", "dossier": {"verdict": "gate2", "reasons": [], "public_summary": ""}}
    try:
        status, reasons, _ = gate2.decide(c, d, signal_from_results(tic, res), lc)
    except Exception as e:  # a catalogue down: record it, the rerun retries it
        return tic, "error", [f"{type(e).__name__}: {e}"]
    return tic, status, reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--down-to", choices=["budget", "all"], default="budget",
                    help="run gate 2 down to Glimmer's-budget cut (default) or to the lowest cut")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import glimmer_value as gv
    from pipeline import fpp, gate2
    from pipeline.paths import DATA, batches
    from pipeline.search import detrend
    from pipeline.vet import load_lc

    rows = gv.survey_rows()
    status = {}
    for b in batches():
        f = DATA / "vet" / f"{b}_v10" / "status.csv"
        if f.exists():
            with open(f, encoding="utf-8") as fh:
                status |= {int(r["tic"]): r["status"] for r in csv.DictReader(fh)}
    tri = {}
    for res in sorted((DATA / "survey_v10").glob("fpp_out*/results.json")) + sorted(a.out.glob("fpp*/results.json")):
        for r in json.loads(res.read_text(encoding="utf-8")):
            if r.get("ran") or r["tic"] not in tri:
                tri[r["tic"]] = r

    def glimmer_final(r: dict) -> bool:
        return r["passes"] and status.get(r["tic"]) == "candidate" and tri_ok(tri.get(r["tic"])) and r["tic"] != KNOWN_BEFORE
    glim = sorted(r["tic"] for r in rows if glimmer_final(r))
    print(f"Glimmer: {sum(r['passes'] for r in rows)} passes of {len(rows)}, {len(glim)} candidates after gate 2 and "
          f"TRICERATOPS: {glim}")

    val, _ = gv.labelled("val")
    yv = np.array([r["planet"] for r in val])
    cut_val = gv.matched_cut(val, np.array([r["glimmer"] for r in val])[yv].mean())
    ok = [r for r in rows if not gv.hard(r["c"], gate2.evidence(r["tic"]))]  # the hard rows bind XGBoost as they bind Glimmer
    keep_all = min((r["rank"] for r in rows if r["tic"] in glim), default=None)
    n_glim = sum(r["passes"] for r in ok)
    budget = sorted((r["rank"] for r in ok), reverse=True)[n_glim - 1]  # XGBoost's top n, as many as Glimmer passed
    cuts = {f"v9 cut {gate2.PASS_RANK}": gate2.PASS_RANK, f"matched on val ({cut_val:.3f})": cut_val,
            f"Glimmer's budget, its top {n_glim} ({budget:.3f})": budget,
            f"keeps all of Glimmer's candidates ({keep_all:.3f})": keep_all}
    # gate 2 runs down to the lowest cut asked for (--down-to budget by default: the keep-all cut passes ~3x as many)
    low = budget if a.down_to == "budget" else min(v for v in cuts.values() if v is not None)

    cache_f = a.out / "gate2.json"
    cache = json.loads(cache_f.read_text(encoding="utf-8")) if cache_f.exists() else {}
    todo = [(r["tic"], r["batch"]) for r in ok if r["rank"] >= low and not r["passes"]
            and (str(r["tic"]) not in cache or cache[str(r["tic"])][0] == "error"
                 or all(w.startswith("no test") for w in cache[str(r["tic"])][1])
                 or any("Timeout" in w or "ConnectionError" in w for w in cache[str(r["tic"])][1]))]  # a catalogue down is no verdict  # the rule before gate2.py's current one
    print(f"gate 2 to run as if XGBoost passed: {len(todo)} signals ({len(cache)} cached)", flush=True)
    with ProcessPoolExecutor(a.jobs) as pool:
        for i, f in enumerate(as_completed([pool.submit(as_pass, j) for j in todo]), 1):
            tic, st, why = f.result()
            cache[str(tic)] = [st, why]
            cache_f.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            print(f"  {i}/{len(todo)} TIC {tic}: {st}" + (f" ({'; '.join(why)[:140]})" if why and st != "candidate" else ""),
                  flush=True)

    lc_dir = {name: lc for name, (_, lc) in batches().items()}
    folded = 0
    for r in ok:
        st = cache.get(str(r["tic"]), [None])[0]
        if st != "candidate" or r["tic"] in tri or (a.out / f"fpp_input_{r['tic']}.json").exists():
            continue
        s, raw = r["c"]["signal"], load_lc(r["tic"], lc_dir[r["batch"]])
        try:
            fo = fpp.fold(r["tic"], s["period"], s["t0"], s["dur_h"], s["depth_ppm"], raw,
                          detrend(raw["time"], raw["flux"], raw["sector"]))
        except fpp.Uncovered as e:
            cache[str(r["tic"])] = ["candidate", [f"TRICERATOPS cannot cover it: {e}"]]
            continue
        (a.out / f"fpp_input_{r['tic']}.json").write_text(json.dumps(fo), encoding="utf-8")
        folded += 1
    if folded:
        print(f"\n{folded} new gate 2 candidates folded: run tools/fpp_survivors.py --folded {a.out} --out {a.out / 'fpp'}, "
              f"then this again")

    print()
    for name, cut in cuts.items():
        if cut is None:
            continue
        if cut < low:
            print(f"XGBoost at {name}: not run (gate 2 ran down to {low:.3f}; --down-to all runs it)")
            continue
        passed = [r for r in ok if r["rank"] >= cut]
        g2, final, pending = [], [], []
        for r in passed:
            st = status.get(r["tic"]) if r["passes"] else cache.get(str(r["tic"]), ["not run"])[0]
            if st != "candidate" or r["tic"] == KNOWN_BEFORE:
                continue
            g2.append(r["tic"])
            t = tri_ok(tri.get(r["tic"]))
            (final if t else pending if t is None else []).append(r["tic"])
        print(f"XGBoost at {name}: {len(passed)} passes, {len(g2)} gate 2 candidates, {len(final)} after TRICERATOPS"
              + (f" (+{len(pending)} without a TRICERATOPS result yet)" if pending else ""))
        print(f"  candidates: {sorted(final)}")
        print(f"  shared with Glimmer: {sorted(set(final) & set(glim))}; XGBoost only: {sorted(set(final) - set(glim))}; "
              f"Glimmer only: {sorted(set(glim) - set(final))}" + (f"; awaiting TRICERATOPS: {sorted(pending)}" if pending else ""))


if __name__ == "__main__":
    main()
