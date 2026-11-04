# Tessa's candidates: predictions for observers

Eight planet candidates from Tessa's survey of nearby K and M dwarfs in TESS full-frame images. None is
confirmed: each is a repeating dip that every check Tessa runs could not explain. Each comes with a dated
prediction of its next transits, so one night on a 1 to 2 m telescope can support it or rule it out.

Made 2026-11-04 17:35 UTC by `python tools/observers.py` from the survey's export. The same rows, with every column, are in
[candidates.csv](candidates.csv). Times are BJD_TDB; coordinates are J2000 (Gaia DR3 at epoch 2016).

| TIC | RA, Dec | Tmag | Period (d) | T0 (BJD_TDB) | Duration | Depth | Size | TRICERATOPS | SNR per transit, 1 m / 2 m |
|---|---|---|---|---|---|---|---|---|---|
| [166321785](https://tessa.braveai.solutions/star/166321785) | 13:46:28.7 -39:45:08.5 | 11.15 | 2.4801264 ± 3.2e-06 | 2460365.00476 ± 1.5 min | 1.37 h | 643 ppm | 1.99 R⊕ | likely planet (NFPP 0.00083) | 2.1 / 3.6 |
| [116046773](https://tessa.braveai.solutions/star/116046773) | 17:46:44.6 +36:06:16.2 | 10.36 | 5.6502857 ± 1.2e-05 | 2459490.75767 ± 1.7 min | 2.27 h | 770 ppm | 2.23 R⊕ | likely planet (NFPP 4.3e-06) | 3.7 / 6.2 |
| [71503196](https://tessa.braveai.solutions/star/71503196) | 20:15:30.4 -09:48:00.3 | 13.31 | 1.5699055 ± 9e-06 | 2460466.85763 ± 2.2 min | 1.2 h | 6454 ppm | 2.9 R⊕ | likely planet (NFPP 0.00043) | 9.4 / 18.2 |
| [9066956](https://tessa.braveai.solutions/star/9066956) | 23:46:26.1 -11:47:47.8 | 13.86 | 0.7134481 ± 6.4e-07 | 2459701.95983 ± 0.7 min | 0.79 h | 4661 ppm | 2.72 R⊕ | likely planet (NFPP 1.4e-09) | 4.4 / 8.6 |
| [50099421](https://tessa.braveai.solutions/star/50099421) | 07:27:59.8 -15:55:57.1 | 10.8 | 5.5374921 ± 1.4e-05 | 2459300.03859 ± 2.1 min | 2.6 h | 889 ppm | 2.33 R⊕ | nearby star to rule out (NFPP 0.002) | 4.2 / 7.3 |
| [321290256](https://tessa.braveai.solutions/star/321290256) | 10:36:56.1 -10:55:47.6 | 9.91 | 2.3318812 ± 1.3e-06 | 2459986.3238 ± 0.6 min | 1.8 h | 556 ppm | 2.19 R⊕ | nearby star to rule out (NFPP 0.027) | 2.5 / 4.2 |
| [61220826](https://tessa.braveai.solutions/star/61220826) | 15:10:23.1 -29:08:22.4 | 10.72 | 11.428717 ± 0.00011 | 2459685.90774 ± 5.1 min | 3.15 h | 1103 ppm | 2.65 R⊕ | nearby star to rule out (NFPP 0.0016) | 5.8 / 10.1 |
| [79490453](https://tessa.braveai.solutions/star/79490453) | 21:26:44.8 -50:12:30.0 | 14.04 | 8.3743864 ± 5e-05 | 2458570.50875 ± 2.9 min | 1.7 h | 8121 ppm | 2.35 R⊕ | nearby star to rule out (NFPP 0.0082) | 10.4 / 20.4 |

"likely planet" means TRICERATOPS gives NFPP under 0.001 (a nearby star is very unlikely to be the source);
"nearby star to rule out" means FPP is under 0.5 but a nearby star is not yet ruled out, so high-resolution imaging
or on-target photometry with a sharper camera is the first test.

## Next windows

**TIC 166321785** (likely planet; timing error now ±2.3 min; 49 usable ground nights in the coming year); preprint [doi:10.5281/zenodo.23957970](https://doi.org/10.5281/zenodo.23957970)

- 2027-01-15 23:42 to 2027-01-16 02:09 UTC, SAAO, South Africa (alt 44°, ±2.4 min)
- 2027-02-12 06:24 to 08:51 UTC, Cerro Tololo, Chile (alt 70°, ±2.5 min)
- 2027-02-17 05:26 to 07:53 UTC, Cerro Tololo, Chile (alt 63°, ±2.5 min)
- Refuted if: high-quality observations covering the predicted windows from 2027-01-15 23:42 to 2027-01-16 02:09 UTC and from 2027-02-12 06:24 to 2027-02-12 08:51 UTC show no 643 ppm dip within 2.4 to 2.5 min of prediction, or if the dip depth changes with colour, indicating a blended binary.

**TIC 116046773** (likely planet; timing error now ±6.1 min; 10 usable ground nights in the coming year); preprint [doi:10.5281/zenodo.23957956](https://doi.org/10.5281/zenodo.23957956)

- 2027-04-12 08:02 to 11:31 UTC, McDonald, Texas (alt 69°, ±6.6 min)
- 2027-04-29 06:50 to 10:19 UTC, McDonald, Texas (alt 68°, ±6.6 min)
- 2027-05-16 05:38 to 09:08 UTC, McDonald, Texas (alt 67°, ±6.7 min)
- TESS sector 117, 2027-05-10 to 2027-06-06 (±6.7 min)
- Refuted if: sensitive observations covering the predicted time within 6.6 min show no 770 ppm dip lasting 2.3 h, or if the dip depth changes with colour, indicating a blended binary.

**TIC 71503196** (likely planet; timing error now ±7.6 min; 98 usable ground nights in the coming year); preprint [doi:10.5281/zenodo.23957922](https://doi.org/10.5281/zenodo.23957922)

- 2027-04-24 17:04 to 19:34 UTC, Siding Spring, Australia (alt 56°, ±9 min)
- 2027-04-26 06:44 to 09:14 UTC, Cerro Tololo, Chile (alt 47°, ±9 min)
- 2027-05-05 16:47 to 19:17 UTC, Siding Spring, Australia (alt 61°, ±9 min)
- TESS sector 116, 2027-04-12 to 2027-05-09 (±9 min)
- Refuted if: data good enough to see the 6454 ppm dip lasting 1.2 hours shows no dip at the predicted time within the timing error of 9 min on those nights, 7.6 min now, or if the dip has a different depth in different colours indicating a blended binary.

**TIC 9066956** (likely planet; timing error now ±2.2 min; 231 usable ground nights in the coming year); preprint [doi:10.5281/zenodo.23957937](https://doi.org/10.5281/zenodo.23957937)

- 2026-11-04 18:26 to 20:18 UTC, SAAO, South Africa (alt 69°, ±2.2 min)
- 2026-11-05 11:33 to 13:25 UTC, Siding Spring, Australia (alt 60°, ±2.2 min)
- 2026-11-06 04:41 to 06:33 UTC, McDonald, Texas (alt 39°, ±2.2 min)
- TESS sector 134, 2028-08-16 to 2028-09-12 (±3.1 min)
- Refuted if: observations good enough to see a 4661 ppm dip lasting 0.8 hours show no dip within 2.2 min of the predicted time, or if the dip depth is different in different colours indicating a blended binary.

**TIC 50099421** (nearby star to rule out; timing error now ±7.9 min; 13 usable ground nights in the coming year); preprint [doi:10.5281/zenodo.23957985](https://doi.org/10.5281/zenodo.23957985)

- 2026-11-20 22:32 to 2026-11-21 02:24 UTC, SAAO, South Africa (alt 63°, ±8 min)
- 2026-11-26 11:26 to 15:18 UTC, Haleakala, Hawaii (alt 53°, ±8 min)
- 2026-12-07 13:13 to 17:05 UTC, Siding Spring, Australia (alt 67°, ±8 min)
- TESS sector 110, 2026-10-30 to 2026-11-26 (±8 min)
- Refuted if: Tessa is wrong if data good enough to see an 889 ppm 2.6 hour dip shows no dip at the predicted time within 7.9 min, or if the dip has a different depth in different colours indicating a blended binary.

**TIC 321290256** (nearby star to rule out; timing error now ±1.2 min; 61 usable ground nights in the coming year); preprint [doi:10.5281/zenodo.23958029](https://doi.org/10.5281/zenodo.23958029)

- 2026-12-22 13:12 to 16:03 UTC, Haleakala, Hawaii (alt 58°, ±1.3 min)
- 2026-12-27 05:07 to 07:58 UTC, Cerro Tololo, Chile (alt 51°, ±1.3 min)
- 2026-12-29 13:05 to 15:55 UTC, Haleakala, Hawaii (alt 58°, ±1.3 min)
- Refuted if: Tessa is wrong if good stacked 2 m data covering the predicted 2026-12-22 13:12 to 2026-12-22 16:03 UTC window within 1.3 min shows no 556 ppm 1.8 hour dip, or if the dip depth changes with colour, a sign of a blended binary.

**TIC 61220826** (nearby star to rule out; timing error now ±23.8 min; 3 usable ground nights in the coming year); preprint [doi:10.5281/zenodo.23957988](https://doi.org/10.5281/zenodo.23957988)

- 2027-04-18 21:24 to 2027-04-19 02:25 UTC, SAAO, South Africa (alt 87°, ±26.1 min)
- 2027-05-11 17:57 to 22:59 UTC, SAAO, South Africa (alt 64°, ±26.4 min)
- 2027-06-15 00:50 to 05:53 UTC, Cerro Tololo, Chile (alt 77°, ±26.9 min)
- TESS sector 115, 2027-03-14 to 2027-04-11 (±25.8 min)
- Refuted if: Tessa is wrong if data good enough to see the 1103 ppm dip, such as 1 m data with SNR 5.8 or 2 m data with SNR 10.1, shows no dip within 23.8 min of the predicted time from the 11.42872-day period, or if the dip has a different depth in different colours, which means a blended binary.

**TIC 79490453** (nearby star to rule out; timing error now ±24.1 min; 11 usable ground nights in the coming year); preprint [doi:10.5281/zenodo.23958000](https://doi.org/10.5281/zenodo.23958000)

- 2027-05-26 05:06 to 08:40 UTC, Cerro Tololo, Chile (alt 50°, ±25.8 min)
- 2027-06-03 14:04 to 17:38 UTC, Siding Spring, Australia (alt 53°, ±25.9 min)
- 2027-06-11 23:03 to 2027-06-12 02:37 UTC, SAAO, South Africa (alt 62°, ±26 min)
- Refuted if: If data good enough to see an 8121 ppm dip covering 2027-05-26 05:06 to 2027-05-26 08:40 UTC shows no dip within 25.8 min of prediction, or the dip depth changes with colour, the candidate is refuted as a blended binary.

Every window for the coming year is on each candidate's page on the site and in its case file
(`data/vet/*_v10/candidates/<tic>.json`, field `when.ground`).
