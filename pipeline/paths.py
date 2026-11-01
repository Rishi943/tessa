import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# TESSA_DATA points the pipeline at another data folder: tools/demo.py runs it on the one-star bundle in demo/data
DATA = Path(os.environ["TESSA_DATA"]).resolve() if os.environ.get("TESSA_DATA") else ROOT / "data"
CACHE = DATA / "cache"
CATALOGS = DATA / "catalogs"



def batches() -> dict[str, tuple[Path, Path]]:
    """Every searched batch: name -> (search results csv, light-curve folder). The survey and batch2, plus any
    data/batchN_results.csv (batch3 is the visitors' search stars, so the next survey batch is batch4)."""
    out = {"survey": (DATA / "survey_results.csv", CACHE / "lc"), "batch2": (DATA / "batch2_results.csv", CACHE / "lc_batch2")}
    for p in sorted(DATA.glob("batch*_results.csv"), key=lambda p: int("0" + "".join(c for c in p.name[5:8] if c.isdigit()))):
        name = p.name.removesuffix("_results.csv")
        out.setdefault(name, (p, CACHE / f"lc_{name}"))
    return out


for d in (DATA, CACHE, CATALOGS):
    d.mkdir(parents=True, exist_ok=True)
