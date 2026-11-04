"""Put Tessa's candidate preprints on Zenodo: one record each, with a DOI printed in its own PDF.

  python tools/zenodo.py create TIC [TIC ...]   a draft per star: reserve the DOI, render the PDF with it, upload the
                                                PDF, transit CSV and parameter JSON, and set the metadata from
                                                preprints/TIC<tic>/zenodo_description.txt
  python tools/zenodo.py check                  read every draft back and compare it with what was sent
  python tools/zenodo.py publish                publish the drafts that pass the check (permanent)

The token is ZENODO_TOKEN in .env (scopes deposit:write, deposit:actions); it is never printed. The drafts' ids and
DOIs are kept in preprints/zenodo.json, so a rerun reuses a draft rather than reserving a second DOI.
"""
import html
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
PRE = ROOT / "preprints"
STATE = PRE / "zenodo.json"
API = "https://zenodo.org/api/deposit/depositions"
RELATION = {"Is supplemented by": "isSupplementedBy", "References": "references"}


def token() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ZENODO_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("no ZENODO_TOKEN in .env")


S = requests.Session()
S.headers["Authorization"] = f"Bearer {token()}"


def state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save(st: dict) -> None:
    STATE.write_text(json.dumps(st, indent=1), encoding="utf-8")


def parse(tic: int) -> dict:
    """The fields of zenodo_description.txt."""
    t = (PRE / f"TIC{tic}" / "zenodo_description.txt").read_text(encoding="utf-8")
    sec = dict(re.findall(r"(?ms)^(TITLE|DESCRIPTION|KEYWORDS|RELATED WORKS)\n(.*?)(?=^\b(?:TITLE|DESCRIPTION|KEYWORDS|RELATED WORKS|RESOURCE TYPE)\b|\Z)", t))
    rel = []
    for block in sec["RELATED WORKS"].strip().split("\n\n"):
        f = dict(line.split(": ", 1) for line in block.strip().splitlines())
        r = {"identifier": f["Identifier"], "relation": RELATION[f["Relation"]], "scheme": "url"}
        if f.get("Resource type") == "Software":
            r["resource_type"] = "software"
        rel.append(r)
    return {"title": sec["TITLE"].strip(), "description": to_html(sec["DESCRIPTION"].strip()),
            "keywords": [k for k in sec["KEYWORDS"].strip().splitlines() if k.strip()], "related": rel}


def to_html(text: str) -> str:
    """Paragraphs and '- ' lists, as Zenodo's description field wants them."""
    out, items = [], []
    for line in text.splitlines():
        if line.startswith("- "):
            items.append(f"<li>{html.escape(line[2:])}</li>")
            continue
        if items:
            out.append("<ul>" + "".join(items) + "</ul>"); items = []
        if line.strip():
            out.append(f"<p>{html.escape(line)}</p>")
    if items:
        out.append("<ul>" + "".join(items) + "</ul>")
    return "".join(out)


def metadata(tic: int, doi: str) -> dict:
    p = parse(tic)
    return {"upload_type": "publication", "publication_type": "preprint", "title": p["title"],
            "creators": [{"name": "Barve, Rushikesh", "affiliation": "Brave AI Solutions"}],
            "description": p["description"], "keywords": p["keywords"], "license": "cc-by-4.0",
            "publication_date": date.today().isoformat(), "related_identifiers": p["related"],
            "prereserve_doi": {"doi": doi}, "access_right": "open"}


def files(tic: int) -> list[Path]:
    d = PRE / f"TIC{tic}"
    return [d / f"TIC{tic}_preprint.pdf", d / f"TIC{tic}_predicted_transits.csv", d / f"TIC{tic}_candidate.json"]


def create(tics: list[int]) -> None:
    st = state()
    for tic in tics:
        rec = st.get(str(tic))
        if rec and rec.get("published"):
            print(f"TIC {tic}: already published, {rec['doi']}"); continue
        if not rec:
            r = S.post(API, json={"metadata": {"prereserve_doi": True}}); r.raise_for_status()
            d = r.json()
            rec = {"id": d["id"], "doi": d["metadata"]["prereserve_doi"]["doi"], "bucket": d["links"]["bucket"]}
            st[str(tic)] = rec; save(st)
        # the PDF carries its own DOI and date
        subprocess.run([sys.executable, "-m", "pipeline.preprint", str(tic), "--pdf", "--doi", rec["doi"],
                        "--date", date.today().isoformat()], cwd=ROOT, check=True, capture_output=True)
        have = {f["filename"] for f in S.get(f"{API}/{rec['id']}/files").json()}
        for f in files(tic):
            if f.name in have:
                S.delete(f"{API}/{rec['id']}/files/" + next(x["id"] for x in S.get(f"{API}/{rec['id']}/files").json() if x["filename"] == f.name))
            with open(f, "rb") as fh:
                S.put(f"{rec['bucket']}/{f.name}", data=fh).raise_for_status()
        r = S.put(f"{API}/{rec['id']}", json={"metadata": metadata(tic, rec["doi"])})
        if not r.ok:
            sys.exit(f"TIC {tic}: metadata refused: {r.status_code} {r.text[:600]}")
        print(f"TIC {tic}: draft {rec['id']}, DOI {rec['doi']}", flush=True)


def check() -> list[int]:
    ok = []
    for tic, rec in state().items():
        d = S.get(f"{API}/{rec['id']}").json()
        m, want = d["metadata"], metadata(int(tic), rec["doi"])
        names = sorted(f["filename"] for f in d.get("files", []))
        cand = (PRE / f"TIC{tic}" / f"TIC{tic}_candidate.json").read_text(encoding="utf-8")  # render() puts the DOI here too
        import hashlib
        local = {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in files(int(tic))}
        stale = [f["filename"] for f in d.get("files", []) if local.get(f["filename"]) != f.get("checksum")]
        problems = [k for k in ("title", "upload_type", "publication_type", "license") if
                    (m.get(k, {}).get("id") if isinstance(m.get(k), dict) else m.get(k)) != want[k]]
        if names != sorted(f.name for f in files(int(tic))):
            problems.append(f"files {names}")
        if m.get("prereserve_doi", {}).get("doi") != rec["doi"]:
            problems.append("doi")
        if len(m.get("related_identifiers", [])) != len(want["related_identifiers"]):
            problems.append("related works")
        if [c["name"] for c in m.get("creators", [])] != ["Barve, Rushikesh"]:
            problems.append("creators")
        if rec["doi"] not in cand:
            problems.append("the parameters file lacks the DOI (PDF not rendered with it)")
        if stale:
            problems.append(f"uploaded files differ from local: {stale}")
        print(f"TIC {tic}: {rec['doi']} {'ok' if not problems else 'PROBLEMS ' + ', '.join(problems)}"
              f"{' (published)' if d.get('submitted') else ''}", flush=True)
        if not problems:
            ok.append(int(tic))
    return ok


def publish() -> None:
    st = state()
    for tic in check():
        rec = st[str(tic)]
        if rec.get("published"):
            continue
        r = S.post(f"{API}/{rec['id']}/actions/publish")
        if not r.ok:
            print(f"TIC {tic}: publish refused: {r.status_code} {r.text[:400]}"); continue
        rec["published"] = r.json()["links"]["record_html"] if "record_html" in r.json()["links"] else r.json()["links"].get("html")
        save(st)
        print(f"TIC {tic}: published {rec['doi']} {rec['published']}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cmd = sys.argv[1]
    if cmd == "create":
        create([int(t) for t in sys.argv[2:]])
    elif cmd == "check":
        check()
    elif cmd == "publish":
        publish()
