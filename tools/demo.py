"""Tessa on one star, end to end: the survey's own code, live, on the light curves bundled in demo/data.

  python tools/demo.py                 TIC 71503196, the likely planet XGBoost ranked low and Glimmer passed
  python tools/demo.py --list          the bundled stars
  python tools/demo.py TIC             another bundled star
  python tools/demo.py --replay        Glimmer's saved dossier instead of a live call (no network for that step)

Steps: the light curve (bundled, as fetched from MAST) -> the blind search for repeating dips and gate 1 -> the code
checks, difference imaging and System 1 (XGBoost) -> the measurements Glimmer may ask for -> Glimmer, the fine-tuned
Muse Glimmer 30B, makes the call with tools -> gate 2: catalogue novelty (live), the season test and the next
transits -> TRICERATOPS's false-positive probability from the survey run.

Glimmer is called through https://tessa.braveai.solutions/api/glimmer, which holds the Modal key and rate-limits;
its container sleeps when idle and takes 3-5 minutes to wake. Set GLIMMER_VETTER_URL and GLIMMER_API_KEY to call
your own deployment (serve/glimmer.py) instead. The run works in .demo_run/, a fresh copy of demo/data, so the repo
stays clean; demo/data is made by tools/demo_pack.py from the survey's data.
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "demo" / "data"
RUN = ROOT / ".demo_run" / "data"
DEFAULT_TIC = 71503196
PUBLIC_GLIMMER = "https://tessa.braveai.solutions/api/glimmer"
WAKE_MINUTES = 10

T0 = time.time()


def say(step: str, text: str = "") -> None:
    print(f"\n[{time.time() - T0:6.0f} s] {step}" + (f"\n  {text}" if text else ""), flush=True)


def f(v, spec: str = ".2f", none: str = "n/a") -> str:
    """A number for the screen; checks leave null what they could not measure."""
    return none if v is None else format(v, spec)


def line(text: str) -> None:
    print(f"  {text}", flush=True)


def wake(base: str) -> None:
    """Poll the endpoint's health until Glimmer answers; the first poll is what wakes a sleeping container."""
    url = f"{base.rstrip('/')}/health"
    deadline = time.time() + WAKE_MINUTES * 60
    first = True
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                body = r.read().decode() or "{}"
                ready = r.status == 200 and (json.loads(body).get("ready", True) if body.startswith("{") else True)
        except Exception:
            ready = False
        if ready:
            line("Glimmer is awake.")
            return
        if first:
            line("Waking Glimmer: its GPU container sleeps when idle and takes 3-5 minutes to start ...")
            first = False
        time.sleep(15)
        print(".", end="", flush=True)
    raise SystemExit(f"\nGlimmer did not wake within {WAKE_MINUTES} minutes; run again with --replay to use its saved dossier")


def run(tic: int, replay: bool, data: Path) -> dict:
    """Every step on one star, with the pipeline reading `data` (TESSA_DATA). Returns what each step produced."""
    os.environ["TESSA_DATA"] = str(data)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    from pipeline import gate2, search, tools, vet
    from pipeline.paths import CACHE, DATA

    meta = json.loads((DATA / "demo.json").read_text(encoding="utf-8"))["stars"][str(tic)]
    batch = meta["batch"]
    lc_dir = CACHE / f"lc_{batch}" if (CACHE / f"lc_{batch}").exists() else CACHE / "lc"
    out = {}

    say(f"1. Light curve: TIC {tic}")
    raw = vet.load_lc(tic, lc_dir)
    star = vet.star_info(tic)
    sectors = sorted({int(s) for s in raw["sector"]})
    line(f"{len(raw['time']):,} brightness measurements from TESS sectors {', '.join(map(str, sectors))}")
    line(f"the star: Teff {f(star['teff'], '.0f')} K, radius {f(star['rad'])} Sun, {f(star['distance'], '.0f')} pc away, "
         f"TESS mag {f(star['tmag'], '.1f')}")

    say("2. Search: every orbit from 0.5 days up, for a dip that repeats (CPU; the survey ran this on GPUs)")
    gate1 = json.loads((DATA / f"{batch}_gate1.json").read_text(encoding="utf-8"))
    row, _ = search.search_star(raw["time"], raw["flux"], raw["sector"], star["rad"] or float("nan"),
                                star["mass"] or float("nan"), search.bls_numpy)
    ok, why = search.gate(row, gate1["floor"], gate1["margin"])
    line(f"{row['n_freq']:,} trial periods; best dip every {row['dip_period']:.5f} d, {row['dip_depth_ppm']:.0f} ppm deep, "
         f"SNR {row['dip_snr']:.1f}, {row['dip_transits']} transits")
    line(f"gate 1 ({'pass' if ok else 'fail'}): {why}. The bar, SNR {gate1['floor']}, is where 1 star in 100 passes "
         f"when the same search runs on the light curves flipped upside down.")
    out["search"] = row
    if not ok:
        return out
    sig = {"period": float(row["dip_period"]), "t0": float(row["dip_t0"]), "dur": float(row["dip_dur_h"]) / 24,
           "depth_ppm": float(row["dip_depth_ppm"]), "snr": float(row["dip_snr"])}

    say("3. Code checks, difference imaging and System 1")
    checks = vet.finite(vet.vet(tic, sig, lc_dir))
    (DATA / "vet" / batch).mkdir(parents=True, exist_ok=True)
    (DATA / "vet" / batch / f"{tic}.json").write_text(json.dumps(checks, indent=1, default=float), encoding="utf-8")
    s1 = checks.get("system1") or {}
    line(f"size {f(checks['size']['rp_rearth'])} Earth radii; odd/even differ by {f(checks['odd_even']['diff_sigma'], '.1f')} sigma; "
         f"secondary eclipse SNR {f(checks['secondary']['snr'], '.1f')}; V-shape {f(checks['shape']['v_sigma'], '.1f')} sigma")
    cen = checks.get("centroid") or {}
    line(f"difference imaging: {cen.get('verdict', 'not run')}"
         + (f", {cen['offset_arcsec']:.1f}\" from the target" if cen.get("offset_arcsec") is not None else ""))
    rank = s1.get("planet_rank")
    line(f"System 1 (XGBoost) planet rank {f(rank, '.3f')}: "
         f"{'below' if (rank or 0) < gate2.PASS_RANK else 'above'} the {gate2.PASS_RANK} cut XGBoost used on its own")
    reaches = checks["code_verdict"] == "agent" or (checks["code_verdict"] == "reject" and checks["reasons"]
                                                     and all("System 1" in r for r in checks["reasons"]))
    line(f"code: {checks['code_verdict']}" + (f" ({'; '.join(checks['reasons'])})" if checks["reasons"] else ""))
    out["checks"] = checks
    if not reaches:
        line("Code settled it: the signal never reaches Glimmer.")
        return out
    checks["code_verdict"] = "agent"

    say("4. Measurements Glimmer may ask for (computed now, handed over only if it calls the tool)")
    for name in ("sector_depths_by_source", "rotation", "transit_depths"):
        tools.run(name, tic)
    if star.get("source_id"):
        tools.gaia_params([int(star["source_id"])])
    tools.run("stellar_radius", tic)
    if cen.get("verdict") == "off_target" or (checks["neighbours"].get("plausible_within_1px") and cen.get("verdict") != "on_target"):
        tools.run("centroid_all", tic)
    import survey_v10
    answers = survey_v10.answers(tic)
    line("ready: " + ", ".join(k for k, v in answers.items() if not (isinstance(v, dict) and v.get("not_run"))))

    say("5. Glimmer makes the call (Muse Glimmer 30B, fine-tuned by Tessa)")
    from finetune.pilot import SCORE_NOTE, invented, investigate, parse, prompt_text, tool_specs
    saved = DATA / "dossiers" / f"{tic}.glimmer-vetter-v10.v10.json"
    if replay:
        run_ = json.loads(saved.read_text(encoding="utf-8"))
        d, used = run_["dossier"], run_.get("tools", [])
        line("--replay: Glimmer's saved dossier from the survey run")
    else:
        os.environ.setdefault("GLIMMER_VETTER_URL", PUBLIC_GLIMMER)
        os.environ.setdefault("GLIMMER_API_KEY", "public-demo")  # the public route ignores it; the Worker adds the real one
        wake(os.environ["GLIMMER_VETTER_URL"])
        prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
        user = survey_v10.views([{"tic": tic, "batch": batch}])[tic]
        rec = investigate(prompt, user, tool_specs(prompt), answers, model="glimmer-vetter")
        rec |= {"user": user, "answers": {k: answers[k] for k in set(rec["tools"]) if k in answers}}
        d = survey_v10.lenient(rec["text"], parse)
        if d is None:
            raise SystemExit("Glimmer's answer had no verdict:\n" + rec["text"][:2000])
        d["reasons"] = [{"check": "", "why": "", **r} for r in d["reasons"] if isinstance(r, dict)]
        d = {"open_questions": [], "public_summary": "", **d}
        used = rec["tools"]
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_text(json.dumps({"model": "glimmer-vetter-v10", "prompt": "v10", "raw": rec["text"], "dossier": d,
                                     "invented": invented(rec, prompt), "tools": used}), encoding="utf-8")
        line(f"{rec['tokens']['input']:,} tokens in, {rec['tokens']['output']:,} out, {rec['ms'] / 1000:.0f} s")
    line(f"tools it chose: {', '.join(used) or 'none'}")
    line(f"verdict: {d['verdict']}")
    for r in d["reasons"][:8]:
        line(f"  - {r.get('check')}: {r.get('why')}")
    if d.get("public_summary"):
        line(f"summary: {d['public_summary']}")
    out["dossier"] = d

    say("6. Gate 2: catalogues (live), the season test, the next transits")
    status, reasons, extra = gate2.decide(checks, {"prompt": "v10", "dossier": d}, sig, lc_dir)
    nov = extra.get("novelty") or {}
    if nov:
        line(f"novelty: {'new' if nov.get('new') else 'known' if nov.get('complete') else 'incomplete'}"
             + (f"; {'; '.join(nov.get('notes', [])[:3])}" if nov.get("notes") else ""))
    w = extra.get("when") or {}
    if w.get("next_tess"):
        t = w["next_tess"]
        line(f"next TESS look: sector {t['sector']}, {t['start'][:10]} to {t['end'][:10]}, about {t['transits']} transits")
    if w.get("ground"):
        g = w["ground"][0]
        line(f"next ground window: {g['site']}, {g['from_utc']} to {g['to_utc'][11:]} UTC ({w.get('ground_nights')} nights in 90 days)")
    line(f"gate 2: {status}" + (f" ({'; '.join(reasons)[:300]})" if reasons else ""))
    out["status"] = status

    say("7. TRICERATOPS (false-positive probability; from the survey run, see README to run it live)")
    fpp = json.loads((DATA / "demo.json").read_text(encoding="utf-8"))["stars"][str(tic)].get("triceratops")
    if fpp:
        bar = fpp["FPP"] < 0.5 and fpp["NFPP"] < 0.1
        tier = ("likely planet" if fpp["NFPP"] < 0.001 else "nearby star to rule out") if bar else "fails the bar"
        line(f"FPP {fpp['FPP']:.4f}, NFPP {fpp['NFPP']:.1e}: {tier}")
        out["triceratops"] = fpp
    say("Done", f"TIC {tic}: {status}" + (f", TRICERATOPS {tier}" if fpp else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tic", type=int, nargs="?", default=DEFAULT_TIC)
    ap.add_argument("--replay", action="store_true", help="Glimmer's saved dossier instead of a live call")
    ap.add_argument("--list", action="store_true", help="the bundled stars")
    ap.add_argument("--data", type=Path, help=argparse.SUPPRESS)  # tools/demo_pack.py: run in this folder as it is
    a = ap.parse_args()
    if a.data:
        run(a.tic, a.replay, a.data.resolve())
        return
    meta = json.loads((BUNDLE / "demo.json").read_text(encoding="utf-8"))
    if a.list:
        for t, m in meta["stars"].items():
            print(f"TIC {t}: {m['note']}")
        return
    if str(a.tic) not in meta["stars"]:
        raise SystemExit(f"TIC {a.tic} is not bundled; --list shows the ones that are")
    if RUN.exists():
        shutil.rmtree(RUN)
    shutil.copytree(BUNDLE, RUN)
    run(a.tic, a.replay, RUN)


if __name__ == "__main__":
    main()
