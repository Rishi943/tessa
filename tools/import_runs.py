"""Bring runs made elsewhere (a cloud session, Kaggle) into this machine's data/, ready for the recount and the export.

  python tools/import_runs.py runs/2026-11-02

RUNS_DIR mirrors data/: results and gate 1 files, vet folders, Glimmer's dossiers and raw answers, the tools' evidence,
TRICERATOPS results and the light curves too big for git are not in it. This script
  1. copies every file into data/, never overwriting one that is already there (this machine's own runs win), except
     those runs.json lists under "replace" (notes written after an earlier import);
  2. merges RUNS_DIR/survey_v10/runset.json into data/survey_v10/runset.json, so the recount sees every batch;
  3. downloads the gate 1 passes' light curves of each batch searched on Kaggle (runs.json names the notebooks) into
     data/cache/lc_BATCH, with tools/kaggle_job.py and this machine's kaggle.json;
  4. takes the held-out stars out of the visitors' search (data/site/search/stars.json): the pipeline has searched them.
If the Kaggle API refuses this machine's key, download each notebook's output from its page instead (kaggle.com/code/
braveai/tessa-search-holdout > Output > Download) and pass it: python tools/import_runs.py RUNS_DIR holdout=PATH ...,
where PATH is the zip or the folder it unpacks to.
Then: python tools/glimmer_value.py funnel, python -m pipeline.export, and deploy the site.
"""
import json
import shutil
import zipfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
DATA = ROOT / "data"


def main() -> None:
    runs = Path(sys.argv[1]).resolve()
    meta = json.loads((runs / "runs.json").read_text(encoding="utf-8"))
    replace = set(meta.get("replace", []))  # files regenerated after an earlier import: these do overwrite
    copied = kept = 0
    for src in sorted(p for p in runs.rglob("*") if p.is_file()):
        rel = src.relative_to(runs)
        if rel.name == "runs.json" or rel == Path("survey_v10/runset.json"):
            continue
        dest = DATA / rel
        if dest.exists() and rel.as_posix() not in replace:
            kept += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    print(f"copied {copied} files into data/, kept {kept} this machine already had")

    new = json.loads((runs / "survey_v10" / "runset.json").read_text(encoding="utf-8"))
    path = DATA / "survey_v10" / "runset.json"
    old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    seen = {(r["tic"], r["batch"]) for r in old}
    merged = old + [r for r in new if (r["tic"], r["batch"]) not in seen]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged), encoding="utf-8")
    print(f"run set: {len(old)} signals here + {len(merged) - len(old)} new = {len(merged)}")

    import kaggle_job
    given = dict(a.split("=", 1) for a in sys.argv[2:])  # batch=zip or folder downloaded by hand
    for batch, slug in meta.get("kaggle_passes", {}).items():
        dest = DATA / "cache" / f"lc_{batch}"
        if (dest / "manifest.csv").exists():
            print(f"{batch}: light curves already in {dest}")
            continue
        tmp = DATA / "cache" / f"_dl_{slug}"
        if batch in given:
            src = Path(given[batch])
            if src.suffix == ".zip":
                zipfile.ZipFile(src).extractall(tmp)
                src = tmp
        else:
            print(f"{batch}: downloading {slug}'s output from Kaggle ...", flush=True)
            kaggle_job.output(slug, tmp)
            src = tmp
        passes = next(p.parent for p in src.rglob("manifest.csv") if p.parent.name == "passes")
        dest.mkdir(parents=True, exist_ok=True)
        for f in passes.iterdir():
            shutil.copy2(f, dest / f.name)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"{batch}: gate 1 passes' light curves in {dest}")

    stars = DATA / "site" / "search" / "stars.json"
    if stars.exists():
        held = {int(t) for t in (ROOT / "survey" / "holdout.txt").read_text(encoding="utf-8").split()}
        rows = json.loads(stars.read_text(encoding="utf-8"))
        keep = [r for r in rows if int(r["tic"]) not in held]
        stars.write_text(json.dumps(keep, separators=(",", ":")), encoding="utf-8")
        print(f"visitors' search: {len(rows)} -> {len(keep)} stars (the held-out ones removed)")


if __name__ == "__main__":
    main()
