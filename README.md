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
| Sent to Glimmer, the fine-tuned model | 190; code settled the other 595 |
| Glimmer called a planet | 49 |
| Passed novelty, testability and season checks | 8 |
| Passed TRICERATOPS (FPP < 0.5, NFPP < 0.1) | 7 |
| Likely planets (FPP < 0.5, NFPP < 0.001; Giacalone et al. 2021) | 3: TIC 71503196, 116046773, 166321785 |
| Candidates with a nearby star still to rule out (NFPP 0.001 to 0.1) | 3: TIC 321290256, 50099421, 61220826 |
| Also found again | TIC 458191673, a blind rediscovery from before the hackathon, not counted as a candidate |

Of the 785 dips, 656 were rejected, 46 are flagged with what is left to check, 74 were held by code without reaching
the model, and 1 is a known object. The other 8 are the row above.

For example, **TIC 71503196**: a 2.9 Earth-radius body every 1.57 days around an M dwarf 62 pc away, 36 transits,
FPP 0.009, NFPP 0.0004. An 11.8" neighbour is not yet excluded, and XGBoost ranked it low (0.09): Glimmer passed it
on the evidence and TRICERATOPS backed that call, placing it in the "likely planet" class.

For the other three, TRICERATOPS does not favour a nearby false positive (NFPP below 0.1), but it cannot yet rule one
out. Ground photometry that resolves the neighbours during a transit settles it.

A candidate is not a planet. It is a dip the checks could not explain, on none of the TOI, CTOI, confirmed-planet or
eclipsing-binary lists, that predicts its own next transit closely enough to test.

TIC 321290256 and 61220826 are on no published list we checked. TIC 116046773 and 50099421 are also in
[RAVEN](https://zenodo.org/records/19661443), TIC 166321785 in
[eumemic/exoplanets](https://github.com/eumemic/exoplanets), and TIC 71503196 in the ExoMiner++ 2.0 catalogue: Tessa
reached those independently.

## How well it works

Tessa end to end, with Glimmer making the call behind the code's hard rows and rank floor, against NASA's ExoMiner++
2.0 on the held-out test TOIs both scored (`finetune/exo_bench.py`). Test signals were held back from all tuning and
training, their outcomes were settled by follow-up, and this was the first look at them.

| 70 planets, 94 false positives | Planets passed | False positives passed | Precision |
|---|---|---|---|
| **Tessa (Glimmer v10 makes the call)** | 50 | **3** | **94%** |
| NASA ExoMiner++ 2.0 at its published 0.5 | 55 | 7 | 89% |

Tessa lets through fewer than half of ExoMiner++'s false positives, and every pass comes with a dossier that says
why. It works from full-frame data and needs no SPOC data-validation products, so it can vet the 48,945 stars in
its pool that were never 2-minute targets. In the survey, TRICERATOPS, the season
test and the novelty check then work on what Glimmer passes.

Prompt v11 asked Glimmer to start from XGBoost's rank and overrule it only with named evidence. It failed the bar set
before it ran (at least 118 of 149 planets and at most 11 of 177 false positives on dev; it got 107 and 14), so it
was never run on test and v10 stayed.

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
- Glimmer fine-tuned on tool-calling traces from Muse Spark Open, making the planet call with tools, with invented
  numbers caught
- XGBoost System 1 and a faster TRICERATOPS (40 s against 392 s on a laptop, same FPP to 0.0001)
- the website, its 3D map and case files, Ask Tessa, and searches that run in visitors' browsers

No code was carried over. A line-by-line comparison with the old scripts finds 23 of Tessa's 6,850 lines in
common (0.3%), all generic one-liners such as creating a folder or seeding a random generator; no function
was copied (`tools/similarity.py`).

As a test, Tessa re-found TIC 458191673 blind, in normal survey order, at the same period. It counts as a
rediscovery that shows the new pipeline works, not as one of Tessa's candidates.

## How it works

Glimmer, a fine-tuned open-weight Meta model, makes the planet call and picks its own tools. Code can stop a pass,
but it can't make one.

1. **Choose the stars** (`pipeline/targets.py`). Gaia DR3 within 100 pc, K/M colours, matched to the TESS Input
   Catalog; known planet hosts, TOIs, CTOIs and eclipsing binaries removed. 48,945 of them were never TESS
   two-minute targets.
2. **Fetch and search** (`pipeline/fetch.py`, `pipeline/search.py`). Light curves from MAST (SPOC, TESS-SPOC,
   QLP), detrended and searched with a box search on two free Kaggle GPUs. Gate 1 runs the same search on the
   light curve flipped upside down, where no real dip exists, and sets its bar so 1% of stars pass on noise.
3. **Code checks** (`pipeline/vet.py`, `pipeline/centroid.py`). Odd against even depth, secondary eclipse,
   size after dilution, duration for the star, V or flat shape, variability at the period, depth sector by
   sector, Gaia neighbours that could fake the dip, and difference-image centroids from TESS cutouts.
4. **System 1 scores it** (`pipeline/system1.py`). An XGBoost model trained on TESS candidates whose fate follow-up
   settled gives a planet rank. Glimmer gets the rank as one piece of evidence, not as a verdict.
5. **Glimmer makes the call** (`tools/survey_v10.py`, `agent/prompts/vetting.v10.md`, `pipeline/tools.py`). It names
   the live explanations, calls only the tools that separate them (centroid in every sector, depth by data source,
   depth of each transit, Gaia radius, rotation), and returns reject, flag or gate2 with a dossier. Any number it
   writes that is not in its input is caught and counted. Survey tool answers are measured ahead of time and returned
   when it calls.
6. **Gate 2 can stop it** (`pipeline/gate2.py`, `pipeline/novelty.py`, `pipeline/when.py`). Hard rows and a safety
   net hold any pass whose System 1 rank is in the bottom 5%. Code checks the TOI, CTOI, confirmed-planet, TESS EB, VSX,
   Gaia EB and SIMBAD lists, tests that each observing season predicts the others, refits the ephemeris and predicts
   the next TESS sectors and the ground windows at six Las Cumbres Observatory sites, with the SNR one transit would
   give on 0.4, 1 and 2 m telescopes. No testable prediction, no candidate.
7. **TRICERATOPS** (`pipeline/fpp.py`, `pipeline/tri_fast.py`). Scores the chance the dip is a false positive from the
   light curve, the star and its neighbours: FPP below 0.5 and NFPP below 0.1 to stay a candidate.
8. **Notes for people** (`agent/caseclosed.js`, `agent/observing.js`). Muse Spark writes one plain line for
   rejected dips and an observing note for candidates. Muse Wonder 1.0 audited candidates from an earlier run from the
   raw checks; TIC 116046773, 321290256 and 71503196 have no Wonder audit yet.

The ephemeris code was checked against known planets: it predicts GJ 357 b and HD 73583 b within 2 and 4
minutes of their published times.

## Meta models

Built with Muse Spark Open.

| Step | Model | Note |
|---|---|---|
| Teacher: tool-calling traces on about 1,008 training signals | Muse Spark Open (Baseten) | about $32 |
| Fine-tune: 745 traces that reached the true outcome with no invented numbers | Muse Glimmer 30B (QLoRA, one H100) | about $8 |
| The planet call on the survey | Glimmer v10, self-hosted on Modal (`serve/glimmer.py`) | adapter `glimmer-vetter-v10` |
| Case-closed lines and observing notes | Muse Spark 1.3 | |
| Audit of candidates from an earlier run | Muse Wonder 1.0 | |
| Ask Tessa on the site | Muse Spark 1.3 | per question |

Prompts are versioned in `agent/prompts/`; every vetting prompt is kept, so the change from v1 to v10 can be read
in order. XGBoost is not a Meta model: it is a small scorer that gives Glimmer a rank to weigh.

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
python tools/survey_v10.py select               # which signals reach Glimmer
python tools/survey_v10.py tools                # measure the tool answers
python tools/survey_v10.py glimmer              # the planet call (Glimmer on Modal)
python tools/survey_v10.py gate2                # novelty, testability, seasons
python tools/survey_v10_fpp.py data/survey_v10/fpp_in   # fold passes for TRICERATOPS (tools/fpp_survivors.py)
node agent/caseclosed.js data/vet/survey
node agent/observing.js data/vet/survey
python -m pipeline.export                       # data/site for the website

cd site && npm install && npm run dev           # the site; `npx wrangler dev` for Ask Tessa
```

Each module's docstring gives its exact arguments. Tests: `python -m pytest tests`.

## Layout

- `pipeline/`: target selection, fetching, search, vetting, centroids, tools, gate 2, predictions, export
- `agent/`: the Muse model calls and the prompts
- `finetune/`: System 1, the traces, the fine-tune, the val and test gates, the ExoMiner++ benchmark
- `serve/`: Glimmer on Modal
- `tools/`: the v10 survey run, TRICERATOPS runs, benchmarks
- `site/`: the website (Vite, Three.js, no framework) and the Cloudflare Worker behind Ask Tessa
- `survey/holdout.txt`: 1,000 stars drawn at random before any search and never searched, so the candidate
  rate can be tested on stars the method was not tuned on
- `tests/`

## Data

NASA TESS light curves via MAST; ESA Gaia DR3; the TESS Input Catalog v8.2; ExoFOP TOI and CTOI lists; the NASA
Exoplanet Archive; the TESS eclipsing binary catalogue; AAVSO VSX; SIMBAD; TESScut for difference images.
Other searches' candidate lists, for credit: eumemic/exoplanets, RAVEN (Zenodo 19661443), T16 (Roth et al. 2026),
ExoMiner++ (Valizadegan et al. 2025, 2026). Data files are not in the repo; the pipeline downloads them.

This research has made use of the NASA Exoplanet Archive, which is operated by the California Institute of
Technology, under contract with the National Aeronautics and Space Administration under the Exoplanet
Exploration Program.

Built for the Meta AI Developer Hackathon, October to November 2026.
