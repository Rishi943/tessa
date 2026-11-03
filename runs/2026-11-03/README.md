# Runs of 2 to 3 November 2026: batches 6, 7 and 8

27,552 stars that were TESS 2-minute targets, with at least two full-frame-image sectors beyond their 2-minute data
(so the search sees data the 2-minute pipeline did not combine), split in three by rank. The visitors' batch 3 and the
held-out stars are left out. Fetched and searched on Kaggle (tessa-fetch-batchN, tessa-search-batchN), then vetted in a
cloud session.

| | batch 6 | batch 7 | batch 8 |
|---|---|---|---|
| Stars searched | 9,184 | 9,184 | 9,184 |
| Gate 1 bar (SNR, from the flipped light curves) | 14.3 | 14.0 | 15.2 |
| Passes, and passes on the flipped light curves | 311, 91 | 367, 91 | 292, 91 |
| Signals that reach Glimmer (survey_v10 select) | 91 | 121 | 71 |

Glimmer v10 had answered 75 of the 283 when this was saved; `python tools/survey_v10.py glimmer` resumes, then gate 2
and TRICERATOPS. Bring into data/ with `python tools/import_runs.py runs/2026-11-03`. Light curves are not here; the
import downloads the gate 1 passes' light curves from the Kaggle notebooks named in runs.json.
