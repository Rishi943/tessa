# Tessa

Tessa is an AI agent that hunts for planets in NASA TESS data on red stars near the Sun. It explains every
dip it can, writes down why, and turns what is left into predictions anyone with a telescope can test.

**Demo video (3 min):** https://youtu.be/D6WHNySGhhU

**Live site:** https://tessa.braveai.solutions: every star in the survey on a 3D map, a case file for every
dip, and Ask Tessa, where Muse Spark answers questions from the survey's own files.

**Quick links:** [Quickstart: hosted, or one star from this repo](#quickstart) ·
[How it compares with NASA's ExoMiner++](#what-glimmer-adds) ·
[Glimmer v10 weights on Hugging Face](https://huggingface.co/rushi-brave/glimmer-vetter-v10) ·
[Meta models used](#meta-models) · [Predictions for observers](predictions/README.md)

## What it found so far

| | |
|---|---|
| Stars in the pool | 99,461 K and M dwarfs within 100 parsecs with TESS data |
| Searched | 76,318: every one of the pool's 48,945 stars that never had 2-minute data (except 179 with no usable light curve), and 27,552 that did |
| Repeating dips found | 2,698, with each batch's bar set on upside-down light curves so about 1 star in 100 passes on noise (749 pass there) |
| Sent to Glimmer, the fine-tuned model | 715 (714 replies could be read; TIC 178829006 came back twice as broken JSON, so it has no call); code settled the other 1,983 |
| Glimmer called a planet | 159 |
| Passed novelty, ephemeris and season checks | 11 |
| **New candidates** (passed TRICERATOPS: FPP < 0.5, NFPP < 0.1) | **8** |
| Likely planets (FPP < 0.5, NFPP < 0.001; Giacalone et al. 2021) | 4: TIC 71503196, 116046773, 166321785, 9066956 |
| Candidates with a nearby star still to rule out (NFPP 0.001 to 0.1) | 4: TIC 321290256, 50099421, 61220826, 79490453 |
| Also found again | TIC 458191673, a blind rediscovery of a candidate published before the hackathon, not counted among the 8 |

The search ran in eight batches: the survey (4,997 stars), batch 2 (17,980), the 1,000 held-out stars (below), and
batches 4 and 5 (12,449 and 12,341), which finished the never-observed part of the pool on 2 November
(`runs/2026-11-02`); then batches 6 to 8 (9,184 each), stars that were 2-minute targets, searched in their full-frame
data (`runs/2026-11-03`). Of the 2,698 dips, 2,239 were rejected, 158 are flagged with what is left to check, 284 are
set aside (283 by code before the model, almost all with a System 1 rank in its bottom 5%; 1 because Glimmer's reply
could not be read), and 9 are known objects; the other 8 are the candidates. Of the 11 that passed gate 2, 9 clear
TRICERATOPS's bar: TIC 426450210 fails (FPP 1.0) and TIC 376928980 could not be scored (TRICERATOPS placed its stars
in only 3 of 4 sectors), so the site counts both as flagged, and TIC 458191673 (found before the hackathon) as known.
Every count is recomputed from the saved runs by `python tools/glimmer_value.py funnel`.

Two candidates from an earlier run on the site, TIC 240488148 and 316447391, are no longer candidates: both fail the
season test (hide one observing season and a blind search of the others must find the same period), and TRICERATOPS
rates TIC 240488148 a likely nearby false positive (NFPP 0.12).

The newest, from batch 4: **TIC 9066956**, a 2.7 Earth-radius body every 0.713 days (17 hours) around an M dwarf
(3,300 K) 90 pc away, 174 transits over six sectors, on target in difference imaging, every season predicting the
others, FPP 0.050 and NFPP 1.4e-9. The PHANTOM search posted it on Zenodo on 23 October, days before the hackathon;
Tessa found it on its own. Its short period gives 231 nights in the coming year when a ground
telescope can catch a transit. All eight, with coordinates, ephemerides and their next windows, are in one table for
observers: [predictions/README.md](predictions/README.md) (and as CSV).

For example, **TIC 71503196**: a 2.9 Earth-radius body every 1.57 days around an M dwarf 62 pc away, 36 transits,
FPP 0.009, NFPP 0.0004. An 11.8" neighbour is not yet excluded, and XGBoost ranked it low (0.09): Glimmer passed it
on the evidence and TRICERATOPS backed that call. Its FPP and NFPP are below TRICERATOPS's statistical-validation
thresholds (0.015 and 0.001); validation would also need high-resolution imaging, which it does not have yet.

For the other four, TRICERATOPS does not favour a nearby false positive (NFPP below 0.1), but it cannot yet rule one
out. Ground photometry that resolves the neighbours during a transit settles it.

A candidate is not a planet. It is a dip the checks could not explain, on none of the TOI, CTOI, confirmed-planet or
eclipsing-binary lists, that every observing season predicts, with its next transits listed so anyone can test it.

TIC 321290256, 61220826 and 79490453 are on no published list we checked. The other five were also found by other
searches, and Tessa reached each independently: TIC 9066956 by
[PHANTOM](https://doi.org/10.5281/zenodo.23584959), TIC 116046773 by
[Reckling](https://doi.org/10.5281/zenodo.23664505) and in [RAVEN](https://zenodo.org/records/19661443), TIC 50099421
by [Ibrahim](https://doi.org/10.5281/zenodo.23569126) and in RAVEN, TIC 166321785 in
[eumemic/exoplanets](https://github.com/eumemic/exoplanets), and TIC 71503196 in the ExoMiner++ 2.0 catalogue.

Each of the eight has a preprint on Zenodo, with its tests, TRICERATOPS result and every predicted transit to
June 2027: [TIC 71503196](https://doi.org/10.5281/zenodo.23957922) · [TIC 9066956](https://doi.org/10.5281/zenodo.23957937) · [TIC 116046773](https://doi.org/10.5281/zenodo.23957956) · [TIC 166321785](https://doi.org/10.5281/zenodo.23957970) · [TIC 50099421](https://doi.org/10.5281/zenodo.23957985) · [TIC 61220826](https://doi.org/10.5281/zenodo.23957988) · [TIC 79490453](https://doi.org/10.5281/zenodo.23958000) · [TIC 321290256](https://doi.org/10.5281/zenodo.23958029).

## Quickstart

**Hosted:** https://tessa.braveai.solutions. The 3D map of every star in the survey, a case file for every dip (its
light curve, every check, Glimmer's dossier, TRICERATOPS, and for candidates the next transits), Ask Tessa, and
*Search a star*, which runs Tessa's search in your browser on a star nobody has searched.

What runs where on the hosted site:

| Step | Where it runs |
|---|---|
| The map, case files, every number on the site | Static files exported from the pipeline's runs (`pipeline/export.py`); they change only when we export and deploy |
| Ask Tessa | Live: Muse Spark 1.3 through the Meta API, from a Cloudflare Worker; it reads those files and the visitors' searches |
| *Search a star*: the search for a repeating dip and gate 1 | Live, in the visitor's browser (web workers on its CPU), on a star nobody has searched |
| Everything after gate 1 for a visitor's star: code checks, Glimmer, gate 2, TRICERATOPS | Not on the site. A star that clears the bar is recorded with the searcher's name; we run the rest of the pipeline on it by hand, then export and deploy. The visitor may leave an email to hear back; a candidate's page credits them |

The site does not run the whole pipeline: only the search does, in the browser. Running the code checks there too
(the same Python, through WebAssembly) is the next step; Glimmer's call and TRICERATOPS would stay on a server.

**From this repo, on one star.** Python 3.12 or newer and Node 22 or newer; about 10 minutes, most of it the search
on CPU. No keys and no sign-in needed. Every step runs live on your machine except Glimmer's call, which is a
**labelled replay** of its recorded answer from the survey run (see step 4).

```sh
git clone https://github.com/Rishi943/tessa && cd tessa
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows; on Linux and macOS use .venv/bin/ in these three lines
npm ci                                            # Node packages for the model's input view
.venv/Scripts/python -m pytest                    # 24 tests, offline, about 25 s
.venv/Scripts/python tools/demo.py                # TIC 71503196, end to end, about 5 minutes
```

`tools/demo.py` runs the survey's own code on TIC 71503196, whose light curves, cutouts and catalogue rows ship in
`demo/data`:

1. **Search** (about 1 minute on CPU): 221,766 trial periods; it finds the dip every 1.56991 days and clears gate 1.
2. **Code checks, difference imaging and System 1**: XGBoost ranks it 0.196, below the 0.32 cut XGBoost used on its
   own. The survey's rank was 0.090: there, difference imaging had not run when System 1 scored the star (its
   centroid input was missing), and the GPU search fitted the transit a little differently. Same model, same verdict.
3. **The five measurements** Glimmer may ask for, computed from the bundled data.
4. **Glimmer makes the call**: the fine-tuned Muse Glimmer 30B. By default the demo **replays** its recorded answer
   for this star from the survey run, and says so on screen: the tools it chose and its verdict with a dossier. To
   call it live, serve the [published weights](https://huggingface.co/rushi-brave/glimmer-vetter-v10) on your own
   Modal account (one L40S GPU, billed by the minute and scaled to zero after 5 idle minutes):

   ```
   modal secret create glimmer-key GLIMMER_API_KEY=<any string you choose>
   modal run serve/glimmer.py::download      # once: the base model, RedHatAI/Muse-Glimmer-30B-FP8-block (33 GB)
   modal run serve/glimmer.py::adapter --repo rushi-brave/glimmer-vetter-v10   # once: the fine-tuned adapter
   modal deploy serve/glimmer.py             # prints two URLs; use the one for `vetter`
   ```

   Then put `GLIMMER_VETTER_URL=<the vetter URL>` and `GLIMMER_API_KEY=<the same string>` in `.env` and run
   `python tools/demo.py --live`. The first request after a deploy takes 3 to 5 minutes while the GPU wakes.
5. **Gate 2**: the catalogues are queried live, the season test reruns the search with each season hidden (about
   2 minutes), and the next TESS sectors and ground windows are predicted.
6. **TRICERATOPS**, live (about 2 minutes: the star's pixels and every neighbour within 10 pixels from MAST and Gaia,
   then our faster TRICERATOPS) when its own environment exists; TRICERATOPS needs Python 3.13 and leaves out one
   package it never uses: `uv venv --python 3.13 tri-venv`, `uv pip install --python tri-venv --no-deps triceratops==1.1.0`, then
   `uv pip install --python tri-venv pandas matplotlib astropy astroquery pytransit==2.2 mechanicalsoup emcee seaborn numba pyrr lightkurve arviz corner beautifulsoup4 "setuptools<81" pyarrow`.
   Without it, the demo shows the survey run's result and says so. TRICERATOPS is a Monte Carlo calculation, so a fresh
   run differs a little from the survey's (TIC 71503196: NFPP 4.3e-4 in the survey, about 8e-4 in a demo run).

`python tools/demo.py 240488148` shows the other outcome: a candidate from the first run that the season test now
flags. `--list` shows the bundled stars. If CDS is down, put its mirrors
first: `TESSA_VIZIER=vizier.idia.ac.za TESSA_SIMBAD=simbad.harvard.edu`.

**The site, locally:** `cd site && npm ci && npm run dev`. With no `data/site` of your own it downloads the same
export from the hosted site (about 105 MB). Ask Tessa and *Search a star* need the Worker: `npx wrangler dev`, with
`META_API_KEY` in `site/.dev.vars`.

**The whole pipeline in the site's own interface:** `cd site && npm run local` starts the site, the Worker and
`tools/local_run.py` together; open http://localhost:5173/run. *Run a star* streams `tools/demo.py`'s steps onto the
page as they happen: the light curve and fold, the code checks and XGBoost's rank, each tool Glimmer asks for (open
one to see the answer it got back), its verdict and dossier, gate 2, and TRICERATOPS. Badges at the top say what is
live: Glimmer is called live when `.env` has `GLIMMER_VETTER_URL` and `GLIMMER_API_KEY`, and otherwise replays its
recorded answer, labelled as a replay. The hosted site has no such server and says so on that page.

## Meta models

Built with Muse Spark Open.

| Step | Model | Note |
|---|---|---|
| Teacher: tool-calling traces on about 1,008 training signals | Muse Spark Open (Baseten) | about $32 |
| Fine-tune: 745 traces that reached the true outcome with no invented numbers (705 to train, 40 held back to check it) | Muse Glimmer 30B (QLoRA, one H100) | about $8 |
| The planet call on the survey | Glimmer v10, self-hosted on Modal (`serve/glimmer.py`) | adapter `glimmer-vetter-v10`, [weights on Hugging Face](https://huggingface.co/rushi-brave/glimmer-vetter-v10) |
| Case-closed lines and observing notes | Muse Spark 1.3 | |
| A second opinion on every candidate | Muse Wonder 1.0 | about $0.09 a candidate |
| Ask Tessa on the site | Muse Spark 1.3 | per question |

Prompts are versioned in `agent/prompts/`; every vetting prompt is kept, so the change from v1 to v10 can be read
in order. XGBoost is not a Meta model: it is our own scorer, trained during the hackathon, that gives Glimmer a rank to weigh.

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
   depth of each transit, Gaia radius, rotation), and returns reject, flag or gate2 with a dossier. Any number in its reasons that it
   writes that is not in its input is caught and counted. Survey tool answers are measured ahead of time and returned
   when it calls.
6. **Gate 2 can stop it** (`pipeline/gate2.py`, `pipeline/novelty.py`, `pipeline/when.py`). Hard rows and a safety
   net hold any pass whose System 1 rank is in the bottom 5%. Code checks the TOI, CTOI, confirmed-planet, TESS EB, VSX,
   Gaia EB and SIMBAD lists, tests that each observing season predicts the others, refits the ephemeris and predicts
   the next TESS sectors and the ground windows at six Las Cumbres Observatory sites, with the SNR one transit would
   give on 0.4, 1 and 2 m telescopes, a year ahead. A signal whose transits can't be timed is not a candidate.
7. **TRICERATOPS** (`pipeline/fpp.py`, `pipeline/tri_fast.py`). Scores the chance the dip is a false positive from the
   light curve, the star and its neighbours: FPP below 0.5 and NFPP below 0.1 to stay a candidate.
8. **Notes for people** (`agent/caseclosed.js`, `agent/observing.js`). Muse Spark writes one plain line for
   rejected dips and an observing note for candidates. Muse Wonder 1.0 audits every candidate from the raw checks, as a
   second opinion that changes nothing: it passes five and flags three, each for something still open (a neighbour
   within one pixel not ruled out for TIC 71503196 and 321290256; depths that differ between sectors 25 and 26 for
   TIC 116046773).

The ephemeris code was checked against known planets: it predicts GJ 357 b and HD 73583 b within 2 and 4
minutes of their published times.

## How well it works

**On the survey, Glimmer found what XGBoost would have thrown away.** Three of the eight candidates sit below the cut
XGBoost (System 1) used on its own in vetting v9, so XGBoost alone would have discarded them: one of the four likely
planets, and two of the three candidates on no published list (`tools/glimmer_value.py`).

| Candidate | Class | XGBoost rank | XGBoost at its 0.32 cut |
|---|---|---|---|
| TIC 71503196 | likely planet | 0.090 | rejects |
| TIC 321290256 | nearby star to rule out; on no published list | 0.144 | rejects |
| TIC 61220826 | nearby star to rule out; on no published list | 0.131 | rejects |
| TIC 79490453, 116046773, 166321785, 9066956, 50099421 | | 0.443 to 0.932 | passes |

Glimmer is not just a looser cut. Of the 715 survey signals that reached it, it passed 159 and those hold all eight
candidates. Given the same budget, its top 159 signals, XGBoost keeps 5 of the 8. Run in Glimmer's place through gate 2
and TRICERATOPS (`tools/xgb_chain.py`), its passes end with those same 5 and one more, TIC 148894196: a 3.5% eclipse
with a secondary eclipse at SNR 234, a binary that Glimmer rejected and the checks XGBoost used alone in vetting v9
reject too. To keep all eight XGBoost would have to pass 456, nearly three times as many, each one more signal for
gate 2, TRICERATOPS and a person to check.

| XGBoost cut | Survey signals passed | Candidates kept |
|---|---|---|
| 0.32 (vetting v9) | 58 | 5 of 8 |
| 0.144 | 229 | 6 of 8 |
| 0.131 | 257 | 7 of 8 |
| 0.090 | 456 | 8 of 8 |
| **Glimmer v10** | **159** | **8 of 8** |

## What Glimmer adds

XGBoost is a good ranker; on known TOIs, matched to the same recall, it is about as accurate as Glimmer (the table
below). What it can't do is change its mind about a signal for a reason. Glimmer reads the evidence, asks for the
measurements that would separate a planet from each impostor, and writes down why. Three things follow from that.

**It rescues planets XGBoost throws away.** On the held-out test split, XGBoost at its 0.32 cut discards 391 signals,
115 of them real planets. Glimmer, reading the same signals, passes 56 of those planets and 13 of the false positives:
81% of what it rescues is a planet. The per-signal calls are in `runs/benchmarks/test_split.json`;
`python tools/test_bench.py` recomputes these tables from it (`tools/glimmer_misses.py` breaks the misses down). On the survey, the same skill is three of the eight
candidates.

**The fine-tune is what makes it work.** Same server, same prompt, same test signals; only the adapter differs
(`python -m finetune.valgate --model glimmer-base`):

| Test split: 297 planets, 282 false positives | Planets passed | False positives passed | Precision | Planets rescued from XGBoost's 391 rejects |
|---|---|---|---|---|
| Muse Glimmer 30B, base | 102 (34%) | 6 (2.1%) | 94% | 7, for 2 false positives |
| **Glimmer v10, fine-tuned on 745 Spark Open investigations** | **213 (72%)** | 18 (6.4%) | 92% | **56, for 13 false positives** |

The base model mostly agrees with XGBoost and says no; fine-tuning on Muse Spark Open's correct investigations more
than doubles the planets it finds and gives it eight times as many rescues, at almost the same precision.

**Its reasons can be checked.** Every call comes with a dossier: which tools it asked for, what they returned, and why
that settles it. A number in its reasons that is not in its input is caught and counted. A person can read why Glimmer
passed TIC 71503196 (on target in every sector, flat-bottomed, the size the star's density predicts) and disagree
with a specific step, which a rank of 0.09 doesn't allow.

**On known TESS candidates.** The held-out test split (signals held back from training and tuning, outcomes settled
by follow-up), with Tessa against XGBoost alone, its cut set on val to recover as many planets as Glimmer:

| 297 planets, 282 false positives | Planets passed | False positives passed | Precision |
|---|---|---|---|
| Tessa (Glimmer v10 makes the call) | 213 (72%) | 18 (6.4%) | 92% |
| XGBoost in Glimmer's place, cut matched on val | 211 (71%) | 10 (3.5%) | 95% |
| XGBoost at its v9 cut (0.32) | 182 (61%) | 6 (2.1%) | 97% |

On the 164 test signals NASA's ExoMiner++ 2.0 also scored (its score is the mean over its 2-minute and full-frame
runs; Tessa uses full-frame data only):

| 70 planets, 94 false positives | Planets passed | False positives passed | Precision |
|---|---|---|---|
| Tessa (Glimmer v10 makes the call) | 50 | 3 | 94% |
| XGBoost, cut set on dev (val plus the training signals Glimmer never trained on) | 55 | 4 | 93% |
| NASA ExoMiner++ 2.0 at 0.5 | 55 | 7 | 89% |

Against ExoMiner++, Tessa's lower false-positive rate holds up: the 95% paired bootstrap interval for the difference
is -8.5 to -1.1 points, so it excludes zero. Its 5 fewer planets do not (-20 to +5.7 points). Printed by
`python -m finetune.exo_bench`, which also reports val and train separately; only test is clean.

Both scorers in Tessa are ours. System 1 is an XGBoost model trained during the hackathon on the same labelled TOIs
as Glimmer (`finetune/system1.py`): trees chosen by cross-validation grouped by host star, so no star is in two folds;
a calibration fitted on val; test scored once. Its inputs are the code checks plus a transit fit, Gaia's binarity signs
and difference imaging. On the test TOIs ExoMiner++ also scored, it ranks planets above false positives better than
NASA's ExoMiner++: AUC 0.963 against 0.945.

So the division of labour is deliberate: XGBoost's rank goes into Glimmer's input and sets a floor under its passes,
and Glimmer makes the call, explains it, and is willing to pass a signal the rank undervalues. On known TOIs that
willingness costs it about 8 extra false positives in 282 at near-equal recall; on the survey it is where three of the eight candidates
came from, and TRICERATOPS, the season test and the novelty check then work on what it passes. Tessa needs no SPOC
data-validation products, so it can vet the 48,945 stars in its pool that were never 2-minute targets, and search the
others in their full-frame data.

Glimmer v10 and its rank floor were frozen on train and val before they ran on test. The XGBoost comparisons above
were later looks at the same test split, after the v10 run; nothing was changed because of them.

Prompt v11 asked Glimmer to start from XGBoost's rank and overrule it only with named evidence. It failed the bar set
before it ran (at least 118 of 149 planets and at most 11 of 177 false positives on dev; it got 107 and 14), so it
was never run on test and v10 stayed.

## The held-out test

Before any search, 1,000 of the survey's fetched stars were drawn uniformly at random across its ranking and set
aside (`pipeline/holdout.py`); the list was committed then, which timestamps it (`survey/holdout.txt`). Nothing in
Tessa was tuned on them. On 1 November, with everything frozen (the search, the
gate 1 bar, System 1, Glimmer v10 and its prompt, gate 2), they were searched once:

| Held-out stars | |
|---|---|
| Searched (999 had data) | 999 |
| Passed gate 1 at the survey's bar | 23; the same search on the flipped light curves passes 9, the noise expected |
| Rejected by code (binaries, spotted stars, System 1) | 17 |
| Set aside (System 1 rank in its bottom 5%, so not sent to Glimmer) | 1 |
| Sent to Glimmer | 5: it rejected 2 whose dips repeat at the star's spot period, and passed 3 |
| Candidates after gate 2 | **0**: one failed the season test, two were too shallow to time per season, so no testable prediction |

The survey's rate, 8 candidates in 76,318 stars, predicts about 0.10 in 1,000 stars, so none is the most likely
outcome (about 90%). What the held-out stars do show is that every stage behaved on stars it had never seen as it did
on the survey: noise passes in the expected share, code removed the binaries and spotted stars, Glimmer rejected
spot signals for that reason, and gate 2 refused what it could not test. Tessa did not make candidates out of noise. Run again under the final gate 2 (an ephemeris that can be fitted, the
season test), the held-out stars still give no candidate.

## Built before and during the hackathon

**Before:** hand-written research scripts, a separate project run one step at a time, found one planet
candidate, TIC 458191673.01 (2.57-day orbit around a K dwarf 66 pc away). A preprint and a TESS observing request
followed.

**During:** Tessa, built from scratch in this repo (first commit 26 October 2026, 23:02 IST, an hour after the kickoff livestream ended at about 22:00 IST). It is a new
pipeline, not those scripts: agentic, and built to handle far more data. Everything here is new work:

- a target pool of 99,461 nearby K and M dwarfs from Gaia DR3 and the TIC, and light curves fetched in compact
  shards on Kaggle
- a GPU box search over tens of thousands of stars, with a noise floor measured on upside-down light curves
- the vetting checks, difference-image centroids, gate 2 and the predictions of the next transits
- Glimmer fine-tuned on tool-calling traces from Muse Spark Open, making the planet call with tools, with invented
  numbers caught
- System 1, our XGBoost scorer trained on TESS candidates whose fate follow-up settled, and a faster TRICERATOPS (40 s against 392 s on a laptop, same FPP to 0.0001)
- the website, its 3D map and case files, Ask Tessa, and searches that run in visitors' browsers

No code was carried over. A line-by-line comparison with the old scripts finds 23 of Tessa's 6,850 lines in
common (0.3%), all generic one-liners such as creating a folder or seeding a random generator; no function
was copied (`tools/similarity.py`).

As a test, Tessa re-found TIC 458191673 blind, in normal survey order, at the same period. It counts as a
rediscovery that shows the new pipeline works, not as one of Tessa's candidates.

## Also tried: building System 1 from scratch

Inspired by the weight-space programming paper (2 Nov 2026), we asked whether an AI agent could build a replacement
for System 1 (XGBoost) from scratch. We tried both routes on the same 65 inputs, data split and reject cut, and froze
every model before opening the test labels, once. The paper's route, a hand-written model with no training, reached a
test AUC of 0.917 on all 579 test TOIs, short of XGBoost's 0.959 (the 0.963 above is on the subset ExoMiner++ also
scored). AUC is how often a real planet ranks above a false positive; 1.0 is perfect. A neural net designed by an AI
agent and trained from scratch in plain numpy beat XGBoost at 0.969 (+0.010, 95% interval +0.003 to +0.019), and its
probabilities were better calibrated. Rescoring the 2,698 survey dips, it kept all 11 that passed gate 2, the 8
candidates among them, above the 5% floor. Of the 83 dips it would have sent on instead of setting aside, 81 reached
Glimmer, which passed 17; none passed gate 2.

**Why XGBoost is still System 1.** The net needs more work before it can take over:

- At the reject cut its gains are within noise, and it shows no gain on cool (K and M) host stars.
- Glimmer was fine-tuned reading XGBoost's ranks, and the net spreads its ranks differently: the 5% floor would need
  picking again and Glimmer testing again.
- The drivers and similar-signals tools are built from XGBoost's internals.
- Every frozen result we report would change.
- The test labels have now been used once, so any further tuning needs a fresh test set, such as TOIs that got their
  verdicts after our data snapshot.

Code, results and how to rerun: `research/scratch-model/README.md`.

## Run the whole survey

Python 3.12 and Node 22 or newer. This needs the full data (`data/` is built by these steps, not shipped), GPUs
for the search, and Glimmer deployed on Modal.

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
- `survey/holdout.txt`: 1,000 stars drawn at random before any search and set aside, then searched once with
  everything frozen, so the candidate rate could be tested on stars the method was not tuned on
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
