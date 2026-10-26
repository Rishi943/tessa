from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = DATA / "cache"
CATALOGS = DATA / "catalogs"

for d in (DATA, CACHE, CATALOGS):
    d.mkdir(parents=True, exist_ok=True)
