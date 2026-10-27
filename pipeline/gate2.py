"""Gate 2: does a vetted signal become a candidate?

Fixed rules the agent cannot change. A candidate needs every code check in the planet-like range
(the same table the agent reads) or settled by evidence from the agent's tools (pipeline/tools.py,
measured by code), the agent's own "gate2" verdict, nothing in novelty check B, and a test: a
future TESS sector or a night on the ground when the transit time is still known. The agent can
stop a signal by rejecting or flagging it, but never pass one the rules don't.

Run: python -m pipeline.gate2 VET_DIR --results results.csv [--lc data/cache/lc]
     [--model muse-spark-1.3-contributor] [--prompt v5]
Writes VET_DIR/status.csv, and for each candidate VET_DIR/candidates/TIC.json with the novelty
check and the predictions.
"""
import argparse
import csv
import json
from pathlib import Path

from .novelty import novelty
from .paths import CACHE, DATA
from .vet import CENTROID_REJECT_SIGMA
from .when import when


def evidence(tic: int) -> dict:
    path = DATA / "evidence" / f"{tic}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def outside_planet_range(c: dict, ev: dict | None = None) -> list[str]:
    """The checks whose numbers fall outside the planet-like column of the agent's table, unless
    evidence from the agent's tools (pipeline/tools.py, measured by code) settles them."""
    ev = ev or {}
    by_source = ev.get("sector_depths_by_source", {}).get("verdict")
    gaia_duration = [o["duration_ratio"] for o in ev.get("stellar_radius", {}).get("options", [])
                     if o["radius_from"].startswith("Gaia")]
    all_sectors = ev.get("centroid_all", {}).get("verdict")
    out = []
    # a check that couldn't be measured is null (vet.finite); comparisons skip it, and odd/even says so
    ge = lambda v, x: v is not None and v >= x  # noqa: E731
    if c["odd_even"]["diff_sigma"] is None:
        out.append("odd_even: no even-numbered transit to compare")
    elif c["odd_even"]["diff_sigma"] >= 3:
        out.append(f"odd_even: odd and even transits differ by {c['odd_even']['diff_sigma']:.1f} sigma")
    s = c["secondary"]
    if ge(s["snr"], 5) and s["snr"] >= 1.5 * (s["best_brightening_snr"] or 0):
        out.append(f"secondary: a dip at phase {s['phase']:.2f} with SNR {s['snr']:.1f}")
    if c["sectors"].get("p_constant") is not None and c["sectors"]["p_constant"] < 0.01             and by_source not in ("steady", "differs_by_source"):
        out.append(f"sectors: depth changes between sectors (p = {c['sectors']['p_constant']:.1g})")
    if ge(c["size"]["rp_rearth"], 10):
        out.append(f"size: {c['size']['rp_rearth']:.1f} Earth radii")
    if (c["duration"]["ratio"] or 0) > 1.5 and not any(r <= 1.5 for r in gaia_duration):
        out.append(f"duration: {c['duration']['ratio']:.1f}x the central transit")
    if any(k["harmonic"] is not None for k in c["known"]):
        out.append("known: a catalogued object at this period")
    if ge(c["shape"]["v_sigma"], 3):
        out.append(f"shape: V-shaped at {c['shape']['v_sigma']:.1f} sigma")
    cen = c.get("centroid") or {}
    on_target = cen.get("verdict") == "on_target" or all_sectors == "on_target"
    if not c["neighbours"].get("target_found", True):
        out.append("neighbours: the target is not in Gaia, so blends were not checked")
    elif c["neighbours"].get("plausible_within_1px", 0) and not on_target:
        out.append(f"neighbours: {c['neighbours']['plausible_within_1px']} star(s) within a pixel could fake it, "
                   f"and difference imaging is {cen.get('verdict', 'not run')}")
    if cen.get("verdict") == "off_target" and not on_target:
        out.append(f"centroid: the dip sits {cen['offset_arcsec']:.1f}\" off the target ({cen['offset_sigma']:.1f} sigma)")
    pc = c["phase_curve"]
    if ge(pc["ratio_to_depth"], 0.3) and ge(pc["chi2"], 50):
        out.append(f"phase_curve: the star swings {pc['ratio_to_depth']:.2f}x the dip at this period")
    if (c["refit"]["transits"] or 0) < 3:
        out.append(f"refit: only {c['refit']['transits']} transits")
    return out


def evidence_rejects(c: dict, ev: dict) -> list[str]:
    """Rejections from the tools' evidence: the same kind of rule as the code vetting's."""
    why = []
    cen = ev.get("centroid_all", {})
    if cen.get("on_star") and cen["offset_sigma"] >= CENTROID_REJECT_SIGMA:
        why.append(f"centroid_all: difference imaging on every sector puts the dip on Gaia {cen['on_star']['source_id']}, "
                   f"{cen['on_star']['sep_arcsec']}\" away ({cen['offset_sigma']:.1f} sigma off this star)")
    rot = ev.get("rotation", {})
    pc = c["phase_curve"]
    if rot.get("signal_at_rotation_multiple") and pc["ratio_to_depth"] >= 0.3 and pc["chi2"] >= 50:
        why.append(f"rotation: the star's spots turn every {rot['rotation_d']:.3f} d, the dip repeats at "
                   f"{rot['signal_at_rotation_multiple']:g}x that, and the star swings {pc['ratio_to_depth']:.2f}x the dip")
    return why


def decide(checks: dict, dossier: dict | None, sig: dict, lc_dir: Path) -> tuple[str, list[str], dict]:
    """(status, reasons, extra): rejected, awaiting agent, flagged, known or candidate."""
    if checks["code_verdict"] == "reject":
        return "rejected", checks["reasons"], {}
    ev = evidence(checks["tic"])
    if evidence_rejects(checks, ev):
        return "rejected", evidence_rejects(checks, ev), {}
    if dossier is None:
        return "awaiting agent", [], {}
    d = dossier["dossier"]
    agent = [f"{r['check']}: {r['why']}" for r in d["reasons"]]
    if d["verdict"] == "reject":
        return "rejected", ["agent: " + a for a in agent], {}
    rules = outside_planet_range(checks, ev)
    if rules or d["verdict"] == "flag":
        return "flagged", rules or ["agent: " + a for a in agent], {}
    nov = novelty(checks["tic"], sig)
    if not nov["complete"]:
        return "flagged", ["novelty: " + e for e in nov["errors"]], {"novelty": nov}
    if not nov["new"]:
        return "known", [f"{m['list']} {m['id']} at {m['harmonic']:g}x this period" for m in nov["matches"]], {"novelty": nov}
    test = when(checks["tic"], sig, lc_dir)
    if not test["testable"]:
        return "flagged", ["no test: " + test.get("why", "no TESS sector or ground night with the timing still known")], \
            {"novelty": nov, "when": test}
    return "candidate", [], {"novelty": nov, "when": test, "public_summary": d["public_summary"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vet_dir")
    ap.add_argument("--results", required=True, help="search results.csv with the signals")
    ap.add_argument("--lc", default=str(CACHE / "lc"), help="folder with the lc_NNN.npz shards")
    ap.add_argument("--model", default="muse-spark-1.3-contributor")
    ap.add_argument("--prompt", default="v5")
    ap.add_argument("--investigation", default="inv1", help="investigation dossiers win over --prompt ones")
    args = ap.parse_args()
    from .vet import signal_from_results

    vet_dir = Path(args.vet_dir)
    rows = []
    for path in sorted(vet_dir.glob("[0-9]*.json")):
        checks = json.loads(path.read_text(encoding="utf-8"))
        tic = checks["tic"]
        found = [DATA / "dossiers" / f"{tic}.{args.model}.{v}.json" for v in (args.investigation, args.prompt)]
        dpath = next((f for f in found if f.exists()), None)
        dossier = json.loads(dpath.read_text(encoding="utf-8")) if dpath else None
        sig = signal_from_results(tic, Path(args.results))
        status, reasons, extra = decide(checks, dossier, sig, Path(args.lc))
        rows.append({"tic": tic, "status": status, "reasons": "; ".join(reasons)})
        if status == "candidate":
            (vet_dir / "candidates").mkdir(exist_ok=True)
            (vet_dir / "candidates" / f"{tic}.json").write_text(
                json.dumps({"tic": tic, "signal": checks["signal"], "size": checks["size"], "star": checks["star"], **extra},
                           indent=1, default=float), encoding="utf-8")
        print(f"TIC {tic}: {status}" + (f" ({reasons[0][:100]})" if reasons else ""), flush=True)
    with (vet_dir / "status.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["tic", "status", "reasons"])
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
    print(counts)


if __name__ == "__main__":
    main()
