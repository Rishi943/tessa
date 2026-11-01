import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# TESSA_DATA points the pipeline at another data folder: tools/demo.py runs it on the one-star bundle in demo/data
DATA = Path(os.environ["TESSA_DATA"]).resolve() if os.environ.get("TESSA_DATA") else ROOT / "data"
CACHE = DATA / "cache"
CATALOGS = DATA / "catalogs"

for d in (DATA, CACHE, CATALOGS):
    d.mkdir(parents=True, exist_ok=True)
