# Screen recordings for the demo video

Nine clips. Each fills one placeholder in `video/film` (the slot times below are the British take, `vo/drama-yt-42`;
they shift a little when the voice is re-recorded, so record each clip **longer than its slot**, with 2 s of
handles at each end). Drop the files in `video/film/assets/footage/` with the names below, then they replace the
`ph-*` placeholders.

## Setup, once

- 1920x1080, browser at 100% zoom, full screen (F11), no bookmarks bar, no extensions showing, dark OS theme.
- Recorder: OBS, 1920x1080, 60 fps, CQP/CRF about 18, no mic. Cursor visible, but move it slowly and only to click.
- Hosted site clips (03, 05, 10, 11, 12): https://tessa.braveai.solutions after deploy (accuracy section + D1
  migration 0003 must be live), or `cd site && npm run dev` at localhost:5173 if recording before the deploy.
- Local run clips (06-09): `cd site && npm run local` with `GLIMMER_VETTER_URL` and `GLIMMER_API_KEY` in `.env` and
  `META_API_KEY` in `site/.dev.vars`. Check the badges on http://localhost:5173/run/71503196 read **Glimmer: live
  call** and **Ask Tessa: live**. Wake Glimmer first (one warm-up run, then record the second): a cold GPU start
  adds minutes.
- **Do 06-09 as one take**: one live run of TIC 71503196, start to finish (about 6-7 min). Never cut out the badges
  or the clock; we speed it up in the edit, and the clock shows it was sped up, which is honest. Nothing is labelled
  "recorded" because nothing is.

## The clips

| # | File | Slot | What the voice says | What to record |
|---|---|---|---|---|
| 03 | `03-problem.mp4` | 16.7 s | A planet dims its star; TESS watched the whole sky; dips are faint, most are fakes, too few people check | `/survey`, open **How it works**. Let the tour play its first stops: the dip, then the fold. Don't click through fast; the tour's own pace is right. About 25 s. |
| 05 | `05-map.mp4` | 10.1 s | "So I built Tessa." 76,000 stars, 8 new candidates, a case file anyone can read | The home 3D map (`/`). Start still for 2 s, then one slow drag turning the sky, then click a candidate (amber) so it flies in. About 15 s. |
| 06 | `run.mp4` (one take, 06 to 09) | 16.6 s | "Here's one, live on my laptop." Over 200,000 orbits, a dip every 1.5 days; clears the bar set by the flipped light | `/run/71503196`. Hold 2 s on the page with the **live** badges, click **Run TIC 71503196**, let the search finish: light curve, fold, "Cleared the bar". |
| 07 | (same take) | 22.5 s | Impostors: eclipsing pairs, starspots, a neighbour's light. Code checks. XGBoost ranks it low. "But Tessa doesn't stop there." | The code checks step and the XGBoost rank line. Impostor cards are animated beside it in the edit, so keep the run on the left two-thirds; don't scroll so far that the checks leave the screen. |
| 08 | (same take) | 22.5 s | Muse Glimmer, fine-tuned on 745 Spark Open investigations; it calls its own tools; verdict: a planet about 3 Earths | Glimmer's step: "Glimmer is reading the case", each **Glimmer asked: ...** line appearing, open one of them (click its arrow) to show the real answer, then the verdict with its tokens and seconds, and the summary quote. |
| 09 | (same take) | 17.0 s | TRICERATOPS: could 300 nearby stars fake it? False alarm under 1%. Made 10x faster | TRICERATOPS's step to the end: the FPP and NFPP lines and "likely planet", then the final "candidate" line. Keep the clock in view. |
| 10 | `10-accuracy.mp4` | 18.9 s | How accurate? 579 past signals; right 92% of the time; fewer than half ExoMiner++'s false alarms | `/survey#accuracy`. Scroll slowly from the method steps so the section comes into view and the squares fill in; hold; then a slow scroll down to the ExoMiner++ rows (3 vs 7 dots) and hold there. |
| 11 | `11-join.mp4` | 11.1 s | Join at the URL; search a star in your browser; ask Tessa's guide anything | Hosted site: start a 1-star search from Ask Tessa ("Search some stars for me" chip, then **Just one star**), let it run a few seconds, then ask one short question, e.g. "What is Tessa, in one minute?", and let the answer stream in. About 25 s; we'll cut to fit. |
| 12 | `12-prediction.mp4` | 12.4 s | One of her candidates, a small red star 290 light-years away; its case file lists the nights it should dim again, for the coming year | `/star/9066956`: the top of the case file, then a slow scroll to the prediction (the next transit time). Hold on the list of nights. |

## Checks before recording

- Numbers on screen must match the voice: 76,000 stars and 8 new candidates on the map (update the placeholders once
  batches 6-8 are final, before recording 05), 1.57 d and 2.9 Earth radii for TIC 71503196, FPP under 1%.
- The tour, the accuracy section and the case file must be the deployed versions.
- No keys, `.env` or terminal windows on screen at any point. The local run clips show only the browser.
