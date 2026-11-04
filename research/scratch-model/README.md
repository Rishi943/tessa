# Scratch-model experiment (2026-11-04)

Can a frontier model build, from scratch, a replacement for System 1 (XGBoost) that does its job better?
Same 65 inputs, same split, same calibration (Platt on val) and the same CV+ 2% reject cut as production.
All five models were frozen (work/FROZEN.sha256) before the sealed test labels were opened, once.

| Model | What it is | Test AUC | Planets lost at cut | FPs caught at cut | Brier |
|---|---|---|---|---|---|
| xgb | production System 1 | 0.959 | 3.4% | 76.6% | 0.079 |
| hand | hand-written evidence ledger, no training (hand.py) | 0.917 | 5.4% | 71.6% | 0.120 |
| nn | numpy MLP ensemble, no ML framework (nn.py) | **0.969** | **2.7%** | **78.0%** | **0.070** |
| nn-hybrid | nn + the hand ledger's terms as inputs | 0.970 | 2.7% | 77.7% | 0.068 |
| xgb-hybrid | control: XGBoost + the hand terms | 0.958 | 3.7% | 75.5% | 0.079 |

Paired bootstrap over host stars (bootstrap.py), nn minus xgb, 95% intervals:
- all test: AUC +0.010 [+0.003, +0.019]; FPs caught +1.4 pts [-1.7, +4.6]; planets lost -0.7 pts [-2.2, +0.6]
- agent-bound: AUC +0.016 [+0.003, +0.030]; planets lost -1.2 pts [-2.9, 0.0]
- K/M hosts: AUC +0.004 [-0.014, +0.023]; FPs caught -5.4 pts [-15.4, +5.0] (no clear difference)

Findings: the from-scratch net ranks better than XGBoost (significant) and is better calibrated; at the cut the
gains are within noise. The hand-written model works with zero training but generalises worse (train 0.949 ->
test 0.917). The hybrid's train-CV lead (0.977) was leakage: the ledger was written while reading all of train.

Survey (survey.py, glimmer83.py): the net, scoring the 2,698 survey dips, keeps all 11 gate-2 candidates above the
5% floor (ranks 7 of them higher). At the floor it would set aside 135 more (116 Glimmer had rejected, 19 flagged)
and release 83 XGBoost set aside: 81 reached Glimmer v10 (shown the net's rank), 17 passed, gate 2 made none a
candidate (12 fail the season test, 4 have no ephemeris, 1 has one season). VizieR's CDS and NAO mirrors were
unreachable; IDIA answered after retries (TIC 158203997 in the first retry, the last 4 in gate2_retry.json).

Run order: data.py, explore.py / bins.py / probe*.py (train only), hand.py, nn.py, sweep.py, xgb_control.py,
evaluate.py NAMES [--final], bootstrap.py. Use the project .venv.
