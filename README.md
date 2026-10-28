# Tessa

Tessa is an AI agent that hunts for planets in NASA TESS data on red stars near the Sun. It explains every
dip it can, writes down why, and turns what is left into predictions anyone with a telescope can test.

**Live site:** https://tessa.braveai.solutions: every star in the survey on a 3D map, a case file for every
dip, and Ask Tessa, where Muse Spark answers questions from the survey's own files.

## What it found so far

| | |
|---|---|
| Stars in the pool | 99,461 K and M dwarfs within 100 parsecs with TESS data |
| Searched | 22,978, ranked by how small a planet each could show |
| Repeating dips found | 785, with a gate calibrated on upside-down light curves so about 1 star in 100 passes on noise |
| Explained away | 528, each with a written reason |
| Still unexplained | 247, flagged with what is left to check |
| Known planets and candidates found again | 3 |
| New candidates | 7: TIC 240488148, 50099421, 316447391, 166321785, 32206438, 358533152, 61220826 |
| Model cost | $1.40 for every dossier, investigation, note and audit |

Each candidate has a case file on the site with its checks, its weak spots and when the next transits can be
seen. For example, **TIC 240488148**: a 1.4 Earth-radius body every 1.51251 days around a K dwarf 59 pc away,
depth steady over 6 sectors; a Gaia star 5.4" away is not yet excluded; next TESS look: sector 121, August 2027.

A candidate is not a planet. It is a dip no check could explain, that no catalogue lists, and that predicts its
own next transit closely enough to test.

## Built before and during the hackathon

**Before:** hand-written research scripts, a separate project run one step at a time, found one planet
candidate, TIC 458191673.01 (2.57-day orbit around a K dwarf 66 pc away). A preprint and a TESS observing request
followed.

**During:** Tessa, built from scratch in this repo (first commit 26 October 2026, after the kickoff). It is a new
pipeline, not those scripts: agentic, and built to handle far more data. Everything here is new work:

- a target pool of 99,461 nearby K and M dwarfs from Gaia DR3 and the TIC, and light curves fetched in compact
  shards on Kaggle
- a GPU box search over tens of thousands of stars, with a noise floor measured on upside-down light curves
- the vetting checks, difference-image centroids, gate 2 and the predictions of the next transits
- Meta's models writing a dossier for every dip and investigating flags with tools, with invented numbers caught
- the website, its 3D map and case files, Ask Tessa, and searches that run in visitors' browsers

No code was carried over. A line-by-line comparison with the old scripts finds 19 of Tessa's 6,282 lines in
common, all generic one-liners such as creating a folder or a 3D scene; no function was copied.

As a test, Tessa re-found TIC 458191673 blind, in normal survey order, at the same period. It counts as a
rediscovery that shows the new pipeline works, not as one of Tessa's candidates.

## How it works

Code makes every yes. Meta's models read the evidence, chase what doesn't add up, and explain it in words.

1. **Choose the stars** (`pipeline/targets.py`). Gaia DR3 within 100 pc, K/M colours, matched to the TESS Input
   Catalog; known planet hosts, TOIs, CTOIs and eclipsing binaries removed. 48,945 of them were never TESS
   two-minute targets.
2. **Fetch and search** (`pipeline/fetch.py`, `pipeline/search.py`). Light curves from MAST (SPOC, TESS-SPOC,
   QLP), detrended and searched with a box search on two free Kaggle GPUs. Gate 1 runs the same search on the
   light curve flipped upside down, where no real dip exists, and sets its bar so 1% of stars pass on noise.
3. **Code checks** (`pipeline/vet.py`, `pipeline/centroid.py`). Odd against even depth, secondary eclipse,
   size after dilution, duration for the star, V or flat shape, variability at the period, depth sector by
   sector, Gaia neighbours that could fake the dip, and difference-image centroids from TESS cutouts.
4. **Muse Spark writes a dossier** (`agent/batch.js`, `agent/prompts/vetting.v5.md`). It sees every number from
   the checks and must cite them. It may reject or flag; it cannot pass anything as a planet. Any number it
   writes that is not in its input is caught and counted (`agent/dossier.js`).
5. **Muse Spark investigates its flags** (`agent/investigate.js`, `pipeline/tools.py`). An agent loop with seven
   tools: centroid in every sector, depth by data source, depth of each transit, Gaia radius, rotation, timing,
   catalogue lookup. Up to five calls a star, restricted to the star under review.
6. **Gate 2 decides** (`pipeline/gate2.py`, `pipeline/novelty.py`, `pipeline/when.py`). Code settles or
   rejects flags from the tools' evidence, checks the TOI, CTOI, confirmed-planet, TESS EB, VSX, Gaia EB and
   SIMBAD lists, refits the ephemeris from per-sector transit times and predicts the next TESS sectors and the
   ground windows at six Las Cumbres Observatory sites, with the SNR one transit would give on 0.4, 1 and 2 m
   telescopes. No testable prediction, no candidate.
7. **Notes for people** (`agent/caseclosed.js`, `agent/observing.js`). Muse Spark writes one plain line for
   every rejected dip and an observing note for every candidate. Muse Wonder 1.0 audits each candidate from the
   raw checks without seeing Spark's verdict.

The ephemeris code was checked against known planets: it predicts GJ 357 b and HD 73583 b within 2 and 4
minutes of their published times.

## Meta models

| Step | Model | Cost, first batch |
|---|---|---|
| Dossiers for 34 dips | Muse Spark 1.3 | $0.047 |
| Investigating 32 flags with tools | Muse Spark 1.3 | $0.044 |
| Case-closed lines and observing notes | Muse Spark 1.3 | $0.011 |
| Independent audit of the candidates | Muse Wonder 1.0 | $0.131 |
| Ask Tessa on the site | Muse Spark 1.3 | per question |

Prompts are versioned in `agent/prompts/`; v1 to v4 of the vetting prompt are kept to show how it was tuned on
known planets and binaries before the bulk run.

## Run it

Python 3.12 and Node 22 or newer.

```sh
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # bin/ on Linux and macOS
npm install
echo META_API_KEY=... > .env

python -m pipeline.targets                      # build the pool
python -m pipeline.holdout                      # freeze the 1,000 held-back stars
python -m pipeline.fetch --targets data/survey_order.txt --out data/cache/lc
python -m pipeline.search --lc data/cache/lc --stars data/survey_stars.txt --skip survey/holdout.txt --out data/survey
python -m pipeline.vet --all data/vet/survey --results data/survey_results.csv
node agent/batch.js data/vet/survey             # dossiers
python -m pipeline.gate2 data/vet/survey --results data/survey_results.csv
node agent/investigate.js data/vet/survey       # the agent with tools
node agent/caseclosed.js data/vet/survey
node agent/observing.js data/vet/survey
python -m pipeline.export                       # data/site for the website

cd site && npm install && npm run dev           # the site; `npx wrangler dev` for Ask Tessa
```

Each module's docstring gives its exact arguments. Tests: `python -m pytest tests`.

## Layout

- `pipeline/`: target selection, fetching, search, vetting, centroids, tools, gate 2, predictions, export
- `agent/`: the Muse model calls, the investigator agent and the prompts
- `site/`: the website (Vite, Three.js, no framework) and the Cloudflare Worker behind Ask Tessa
- `survey/holdout.txt`: 1,000 stars drawn at random before any search and never searched, so the candidate
  rate can be tested on stars the method was not tuned on
- `tests/`

## Data

NASA TESS light curves via MAST; ESA Gaia DR3; the TESS Input Catalog v8.2; ExoFOP TOI and CTOI lists; the NASA
Exoplanet Archive; the TESS eclipsing binary catalogue; AAVSO VSX; SIMBAD; TESScut for difference images. Data
files are not in the repo; the pipeline downloads them.

Built for the Meta AI Developer Hackathon, October to November 2026.
