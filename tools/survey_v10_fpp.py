"""Fold the light curves of the survey stars Glimmer v10 passed, ready for fast TRICERATOPS (tools/fpp_survivors.py --folded).

A star is folded when its v10 dossier says gate2 and no HARD_ROWS/FLOOR_RANK row stops it (the vetting pass, before novelty and
seasons, like v9's 15 -> 13 step). Uses the signal's own ephemeris (checks["signal"]); a candidate's gate-2 ephemeris is
the same fit.
Run (project .venv): python tools/survey_v10_fpp.py OUT_DIR
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import fpp, gate2  # noqa: E402
from pipeline.paths import CACHE, DATA, batches  # noqa: E402
from pipeline.search import detrend  # noqa: E402
from pipeline.vet import load_lc  # noqa: E402

LC = {name: lc for name, (_, lc) in batches().items()}


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch in LC:
        for p in sorted((DATA / "vet" / f"{batch}_v10").glob("[0-9]*.json")):
            c = json.loads(p.read_text(encoding="utf-8"))
            tic = c["tic"]
            dp = DATA / "dossiers" / f"{tic}.glimmer-vetter-v10.v10.json"
            if not dp.exists():
                continue
            d = json.loads(dp.read_text(encoding="utf-8"))["dossier"]
            if d["verdict"] != "gate2" or gate2.outside_planet_range(c, gate2.evidence(tic), "v10"):
                continue
            s = c["signal"]
            raw = load_lc(tic, LC[batch])
            lc = detrend(raw["time"], raw["flux"], raw["sector"])
            try:
                f = fpp.fold(tic, s["period"], s["t0"], s["dur_h"], s["depth_ppm"], raw, lc)
            except fpp.Uncovered as e:
                rows.append({"tic": tic, "batch": batch, "covered": False, "why": str(e)})
                continue
            (out / f"fpp_input_{tic}.json").write_text(json.dumps(f), encoding="utf-8")
            rows.append({"tic": tic, "batch": batch, "covered": True})
    (out / "coverage.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"{sum(r['covered'] for r in rows)} folded, {sum(not r['covered'] for r in rows)} uncovered")


if __name__ == "__main__":
    main()
