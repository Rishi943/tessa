# Runs of 2 November 2026: the held-out stars, batch 4 and batch 5

Everything the frozen v10 pipeline wrote for these three batches except the light curves (too big for git: they are
the outputs of the Kaggle notebooks named in `runs.json`). It mirrors `data/`, so `python tools/import_runs.py
runs/2026-11-02` copies it into place, downloads the light curves, and the recount and the export then cover every batch.

- `*_results.csv`, `*_gate1.json`: the GPU search of every star and each batch's gate 1 bar
- `vet/BATCH/`: the code checks, difference imaging and System 1 for each gate 1 pass
- `vet/BATCH_v10/`: the same, as Glimmer saw them, with gate 2's `status.csv` and `candidates/`
- `survey_v10/glimmer/`: every Glimmer v10 answer in full (the tool calls it made and its text); `dossiers/` the parsed verdicts
- `evidence/`: the measurements Glimmer could ask for; `validation/`: the season tests
- `survey_v10/fpp_out3*/`: TRICERATOPS on every signal Glimmer passed
