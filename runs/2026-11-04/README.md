# Runs of 3 to 4 November 2026: batches 6 to 8 finished, and the recount

Made on the laptop after `runs/2026-11-03`:

- Glimmer v10's answers for the rest of batches 6 to 8 (`survey_v10/glimmer`, `dossiers`), with the replies that
  could not be read asked again;
- gate 2 under the current rule (`pipeline/gate2.py`) for every signal that had been flagged only for having no test
  window (`tools/regate_notest.py`), and again for the candidates so their windows run a year ahead;
- TRICERATOPS on the new gate 2 passes (`survey_v10/fpp_out4`, `fpp_out5`);
- Muse Wonder's second opinion on TIC 79490453, and the observing notes for every candidate.

The result: 76,318 stars searched, 2,698 repeating dips, 715 sent to Glimmer, 159 passed by it, 11 through gate 2,
and 8 candidates after TRICERATOPS, plus TIC 458191673 found again. Batches 6 to 8 add no candidates.

Bring into data/ with `python tools/import_runs.py runs/2026-11-04` after the 11-02 and 11-03 folders; the files in
runs.json's "replace" list overwrite the earlier ones. Then `python tools/glimmer_value.py funnel` and
`python -m pipeline.export`.
