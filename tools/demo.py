"""Tessa on one star, end to end: the survey's own code on the light curves bundled in demo/data (Glimmer's step replayed unless --live).

  python tools/demo.py                 TIC 71503196, the likely planet XGBoost ranked low and Glimmer passed
  python tools/demo.py --list          the bundled stars
  python tools/demo.py TIC             another bundled star
  python tools/demo.py --live          call your own Glimmer instead of replaying its recorded answer (see below)

Steps: the light curve (bundled, as fetched from MAST) -> the blind search for repeating dips and gate 1 -> the code
checks, difference imaging and System 1 (XGBoost) -> the measurements Glimmer may ask for -> Glimmer, the fine-tuned
Muse Glimmer 30B, makes the call with tools -> gate 2: catalogue novelty (live), the season test and the next
transits -> TRICERATOPS's false-positive probability from the survey run.

Glimmer's step is a REPLAY by default: its recorded answer for the star from the survey run, labelled as such. Every
other step runs live. --live calls your own Glimmer instead: serve the published weights (serve/glimmer.py) and set
GLIMMER_VETTER_URL and GLIMMER_API_KEY; a sleeping container takes 3-5 minutes to wake. The run works in .demo_run/, a fresh copy of demo/data, so the repo
stays clean; demo/data is made by tools/demo_pack.py from the survey's data.
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "demo" / "data"
RUN = ROOT / ".demo_run" / "data"
DEFAULT_TIC = 71503196
WAKE_MINUTES = 10

T0 = time.time()
EVENTS = False  # --events: also print @@{json} lines, which tools/local_run.py streams to the site's /run page


def ev(**kw) -> None:
    if EVENTS:
        print("@@" + json.dumps(kw, default=float), flush=True)


def say(step: str, text: str = "") -> None:
    ev(type="step", title=step, t=round(time.time() - T0))
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
    # Cloudflare turns away Python's default user agent ("Python-urllib") with a 403, so name ours
    req = urllib.request.Request(url, headers={"User-Agent": "tessa-demo/1.0"})
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                body = r.read().decode() or "{}"
                ready = r.status == 200 and (json.loads(body).get("ready", True) if body.startswith("{") else True)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):  # not a sleeping container: waiting won't help
                raise SystemExit(f"\nGlimmer's endpoint refused the health check ({url}: HTTP {e.code}); "
                                 "run again without --live to replay its recorded answer")
            ready = False
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
    raise SystemExit(f"\nGlimmer did not wake within {WAKE_MINUTES} minutes; run again without --live to replay its recorded answer")


def triceratops_live(tic: int, sig: dict, raw: dict, lc_dir: Path, data: Path, tri: Path) -> dict | None:
    """Fold the transit as the survey did (pipeline/fpp.py), then run tools/fpp_survivors.py on it in tri-venv:
    apertures from MAST, neighbours from Gaia, pixel cutouts, then TRICERATOPS. Streams its log; None if it fails."""
    import subprocess
    from pipeline import fpp as fppmod, search
    work = data / "fpp"
    folded, outdir = work / "in", work / "out"
    folded.mkdir(parents=True, exist_ok=True)
    try:
        d = fppmod.fold(tic, sig["period"], sig["t0"], sig["dur"] * 24, sig["depth_ppm"], raw,
                        search.detrend(raw["time"], raw["flux"], raw["sector"]))
    except fppmod.Uncovered as e:
        line(f"TRICERATOPS can't run on this star's data: {e}")
        return None
    (folded / f"fpp_input_{tic}.json").write_text(json.dumps(d), encoding="utf-8")
    line(f"{d['n_points_unbinned']} points near the transits, in {len(d['time'])} bins; geometry from TESS-SPOC sectors "
         f"{d['geometry_sectors']}")
    ev(type="fpp_start", sectors=d["geometry_sectors"])
    p = subprocess.Popen([str(tri), "-u", str(ROOT / "tools" / "fpp_survivors.py"), "--folded", str(folded), "--out", str(outdir),
                          "--only", str(tic), "--max-minutes", "20"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    for raw_line in p.stdout:
        t = raw_line.rstrip()
        # TRICERATOPS' own log says "validated" and prints its raw result row; Tessa never calls a candidate validated
        if t and not t.startswith("{'machine'") and "Warning" not in t and "validated" not in t and '"covered"' not in t:
            line(t[:220])
    p.wait()
    try:
        r = json.loads((outdir / "results.json").read_text(encoding="utf-8"))[0]
    except Exception:
        r = {}
    if not r.get("ran"):
        line(f"TRICERATOPS didn't finish: {r.get('why', 'see the lines above')}")
        return None
    line(f"TRICERATOPS took {r['wall_s']:.0f} s ({r['setup_s']:.0f} s fetching the pixels and neighbours, {r['calc_s']:.0f} s computing)")
    return {"FPP": r["FPP"], "NFPP": r["NFPP"]}


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
    ev(type="star", tic=tic, n=len(raw["time"]), sectors=sectors, teff=star["teff"], rad=star["rad"], dist_pc=star["distance"],
       tmag=star["tmag"], years=round((raw["time"].max() - raw["time"].min()) / 365.25, 1))

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
    ev(type="search", n_freq=int(row["n_freq"]), period=float(row["dip_period"]), depth_ppm=float(row["dip_depth_ppm"]),
       snr=float(row["dip_snr"]), transits=int(row["dip_transits"]), floor=gate1["floor"], ok=bool(ok))
    if not ok:
        return out
    sig = {"period": float(row["dip_period"]), "t0": float(row["dip_t0"]), "dur": float(row["dip_dur_h"]) / 24,
           "depth_ppm": float(row["dip_depth_ppm"]), "snr": float(row["dip_snr"])}
    if EVENTS:
        from pipeline.export import light_curve
        ev(type="chart", tic=tic, period=sig["period"], t0=sig["t0"], depth_ppm=sig["depth_ppm"], dur_h=sig["dur"] * 24,
           **light_curve(search.detrend(raw["time"], raw["flux"], raw["sector"]), sig))

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
    if meta.get("survey_rank") is not None:
        line(f"  the survey run ranked it {meta['survey_rank']:.3f}, the number on its case file: there, difference imaging had not "
             f"run when System 1 scored it, and the GPU search fitted the transit a little differently (README, Try it)")
    reaches = checks["code_verdict"] == "agent" or (checks["code_verdict"] == "reject" and checks["reasons"]
                                                     and all("System 1" in r for r in checks["reasons"]))
    line(f"code: {checks['code_verdict']}" + (f" ({'; '.join(checks['reasons'])})" if checks["reasons"] else ""))
    out["checks"] = checks
    ev(type="checks", verdict=checks["code_verdict"], reasons=checks["reasons"], rp_rearth=checks["size"]["rp_rearth"],
       odd_even_sigma=checks["odd_even"]["diff_sigma"], secondary_snr=checks["secondary"]["snr"], v_sigma=checks["shape"]["v_sigma"],
       centroid=cen.get("verdict", "not run"), xgb_rank=rank, xgb_cut=gate2.PASS_RANK, reaches=bool(reaches))
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
    tools.run("centroid_all", tic)  # all five, so whichever Glimmer calls has an answer (the survey had them all ready)
    import survey_v10
    answers = survey_v10.answers(tic)
    line("ready: " + ", ".join(k for k, v in answers.items() if not (isinstance(v, dict) and v.get("not_run"))))

    say("5. Glimmer makes the call (Muse Glimmer 30B, fine-tuned by Tessa)")
    from finetune.pilot import SCORE_NOTE, invented, investigate, parse, prompt_text, tool_specs
    saved = DATA / "dossiers" / f"{tic}.glimmer-vetter-v10.v10.json"
    if replay:
        run_ = json.loads(saved.read_text(encoding="utf-8"))
        d, used = run_["dossier"], run_.get("tools", [])
        line("REPLAY: Glimmer's recorded answer from the survey run, not a live call (--live calls your own Glimmer)")
    else:
        if not (os.environ.get("GLIMMER_VETTER_URL") and os.environ.get("GLIMMER_API_KEY")):
            raise SystemExit("--live needs your own Glimmer: serve the published weights (serve/glimmer.py), then set "
                             "GLIMMER_VETTER_URL and GLIMMER_API_KEY. Without --live the demo replays its recorded answer.")
        wake(os.environ["GLIMMER_VETTER_URL"])
        prompt = prompt_text("v10") + "\n\n" + SCORE_NOTE
        user = survey_v10.views([{"tic": tic, "batch": batch}])[tic]
        ev(type="glimmer_start", model="glimmer-vetter-v10", live=True)
        rec = investigate(prompt, user, tool_specs(prompt), answers, model="glimmer-vetter",
                          on_call=lambda name, args, ans: ev(type="tool", name=name, args=args, answer=json.dumps(ans, default=float)[:600]))
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
    ev(type="dossier", replay=replay, tools=used, verdict=d["verdict"], reasons=d["reasons"][:8], summary=d.get("public_summary", ""),
       **({} if replay else {"tokens": rec["tokens"], "seconds": round(rec["ms"] / 1000)}))
    line(f"verdict: {d['verdict']}")
    for r in d["reasons"][:8]:
        line(f"  - {r.get('check')}: {r.get('why')}")
    if d.get("public_summary"):
        line(f"summary: {d['public_summary']}")
    out["dossier"] = d

    say("6. Gate 2: catalogues (live), the season test, the next transits")
    status, reasons, extra = gate2.decide(checks, {"prompt": "v10", "dossier": d}, sig, lc_dir)
    if any(r.startswith("novelty:") for r in reasons) and not (extra.get("novelty") or {}).get("complete", True):
        # a catalogue server didn't answer (CDS has outages): say so, and try once more with its mirrors first
        line(f"a catalogue didn't answer ({reasons[0][:120]}); trying again with the mirrors first ...")
        from pipeline import novelty
        novelty.VIZIER_SERVERS = ("vizier.idia.ac.za", "vizier.nao.ac.jp", "vizier.cds.unistra.fr")
        novelty.SIMBAD_SERVERS = ("simbad.harvard.edu", "simbad.cds.unistra.fr")
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
        line(f"next ground window: {g['site']}, {g['from_utc']} to {g['to_utc'][11:]} UTC ({w.get('ground_nights')} nights in the next year)")
    line(f"gate 2: {status}" + (f" ({'; '.join(reasons)[:300]})" if reasons else ""))
    out["status"] = status
    ev(type="gate2", status=status, reasons=reasons, novelty=nov, next_tess=w.get("next_tess"), ground=(w.get("ground") or [None])[0],
       ground_nights=w.get("ground_nights"))

    tri = ROOT / "tri-venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    recorded = json.loads((DATA / "demo.json").read_text(encoding="utf-8"))["stars"][str(tic)].get("triceratops")
    fpp = None
    if tri.exists():
        say("7. TRICERATOPS, live: is it the star, a binary, or a neighbour? (our faster TRICERATOPS, pipeline/tri_fast.py)")
        fpp = triceratops_live(tic, sig, raw, lc_dir, data, tri)
    if fpp is None and recorded:
        if not tri.exists():
            say("7. TRICERATOPS (the survey run's result; to run it live, make tri-venv as the README says)")
        else:
            line("using the survey run's result instead")
        fpp = recorded
        ev(type="fpp_recorded")
    tier = None
    if fpp:
        bar = fpp["FPP"] < 0.5 and fpp["NFPP"] < 0.1
        tier = ("likely planet" if fpp["NFPP"] < 0.001 else "nearby star to rule out") if bar else "fails the bar"
        line(f"FPP {fpp['FPP']:.4f}, NFPP {fpp['NFPP']:.1e}: {tier}")
        out["triceratops"] = fpp
        ev(type="fpp", fpp=fpp["FPP"], nfpp=fpp["NFPP"], tier=tier, live=fpp is not recorded)
    ev(type="done", status=status)
    say("Done", f"TIC {tic}: {status}" + (f", TRICERATOPS {tier}" if tier else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tic", type=int, nargs="?", default=DEFAULT_TIC)
    ap.add_argument("--live", action="store_true", help="call your own Glimmer (GLIMMER_VETTER_URL, GLIMMER_API_KEY) instead of the replay")
    ap.add_argument("--list", action="store_true", help="the bundled stars")
    ap.add_argument("--events", action="store_true", help=argparse.SUPPRESS)  # tools/local_run.py
    ap.add_argument("--data", type=Path, help=argparse.SUPPRESS)  # tools/demo_pack.py: run in this folder as it is
    a = ap.parse_args()
    global EVENTS
    EVENTS = a.events
    if a.data:
        run(a.tic, not a.live, a.data.resolve())
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
    run(a.tic, not a.live, RUN)


if __name__ == "__main__":
    main()
