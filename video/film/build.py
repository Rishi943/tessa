"""Assemble index.html from the voice's timings: the disclosure, the animated scenes, the app footage (cut.py fits each
recording to its window), the "Real app footage" label over those, and the voice track.

  ../../.venv/Scripts/python build.py [--vo 42]

Rerun after re-voicing: every scene's start and length comes from video/vo/<vo>/timings.json.
"""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIS = 5.0          # the silent disclosure card
CLOSE_LEAD = 3.0   # the close's sky starts this long before "Now the sky gets to answer" ends
END = 179.9        # the film ends here, under the 3:00 limit; the close fills what is left

ap = argparse.ArgumentParser()
ap.add_argument("--vo", default="drama-yt-42")  # the British take
a = ap.parse_args()
t = json.loads((HERE.parent / "vo" / a.vo / "timings.json").read_text(encoding="utf-8"))
sc = {s["id"]: s for s in t["scenes"]}
shutil.copy(HERE.parent / "vo" / a.vo / "full.wav", HERE / "assets" / "vo.wav")
script = {s["id"]: s for s in json.loads((HERE.parent / "script.json").read_text(encoding="utf-8"))["scenes"]}

# scene windows on the film's clock: each runs from its first line to the next scene's first line
ids = [s["id"] for s in t["scenes"]]
start = {i: DIS + sc[i]["start"] for i in ids}
last = t["scenes"][-1]
end_all = END
pred_last = sc["12-prediction"]["lines"][-1]
close0 = DIS + pred_last["start"] + pred_last["dur"] - CLOSE_LEAD
win = {}
for k, i in enumerate(ids):
    s0 = DIS if i == "02-hook" else start[i]
    s1 = start[ids[k + 1]] if k + 1 < len(ids) else end_all
    win[i] = (s0, s1)
win["12-prediction"] = (win["12-prediction"][0], close0)
win["close"] = (close0, end_all)

hosts, holders, labels = [], [], []
def host(cid, s0, s1, z):
    hosts.append(f'      <div id="{cid}" data-composition-id="{cid}" data-composition-src="compositions/{cid}.html" '
                 f'data-start="{s0:.2f}" data-duration="{s1 - s0:.2f}" data-track-index="1" data-track-kind="graphics" '
                 f'data-width="1920" data-height="1080" style="z-index:{z}"></div>')
host("disclosure", 0, DIS, 1)
host("open", *win["02-hook"], 1)
host("close", *win["close"], 1)
from cut import cut
clips = cut({i: win[i] for i in ids if i != "02-hook"})
for i in ids:
    if i == "02-hook":
        continue
    s0, s1 = win[i]
    if i in clips:
        holders.append(f'      <video id="v-{i}" class="clip foot" src="{clips[i]["src"]}" muted playsinline data-start="{s0:.2f}" '
                       f'data-duration="{s1 - s0:.2f}" data-track-index="2"></video>')
        sp = clips[i]["speed"]
        labels.append((s0, s1, "Real app footage" + (f" &middot; up to {sp:.0f}&times; speed" if sp >= 1.5 else "")))
    else:
        holders.append(f'      <div id="ph-{i}" class="clip ph" data-start="{s0:.2f}" data-duration="{s1 - s0:.2f}" data-track-index="2">'
                       f'<b>{i}</b><span>{script[i]["footage"]}</span></div>')
        labels.append((s0, s1, "Real app footage"))
lab = [f'      <div id="label-{k}" class="clip label" data-start="{s0:.2f}" data-duration="{s1 - s0:.2f}" data-track-index="3">{t}</div>'
       for k, (s0, s1, t) in enumerate(labels)]

html = f'''<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1920, height=1080" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.14.2/dist/gsap.min.js"></script>
    <style>
      @font-face {{ font-family: "Archivo"; src: url("assets/fonts/archivo.woff2") format("woff2"); font-weight: 300 700; }}
      @font-face {{ font-family: "Courier Prime"; src: url("assets/fonts/courier-400.woff2") format("woff2"); font-weight: 400; }}
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{ width: 1920px; height: 1080px; overflow: hidden; background: #0b0b0a; }}
      #root {{ position: relative; width: 1920px; height: 1080px; }}
      /* stand-ins for screen recordings not made yet: replace each with a <video> of the same window */
      .ph {{ position: absolute; inset: 0; background: #121211; color: #9C9B95; display: flex; flex-direction: column;
             align-items: center; justify-content: center; gap: 18px; font: 400 30px/1.3 "Courier Prime", monospace; z-index: 2; }}
      .ph b {{ color: #ECEBE6; font: 600 54px/1 Archivo, sans-serif; }}
      .foot {{ position: absolute; inset: 0; width: 1920px; height: 1080px; object-fit: cover; z-index: 2; background: #0b0b0a; }}
      .label {{ position: absolute; right: 48px; bottom: 40px; z-index: 5; color: rgba(236, 235, 230, 0.72);
               font: 400 20px/1 "Courier Prime", monospace; letter-spacing: 0.14em; text-transform: uppercase;
               padding: 9px 14px; border: 1px solid rgba(236, 235, 230, 0.26); border-radius: 4px; background: rgba(11, 11, 10, 0.55); }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{end_all:.2f}" data-width="1920" data-height="1080">
{chr(10).join(hosts)}
{chr(10).join(holders)}
{chr(10).join(lab)}
      <audio id="vo" class="clip" src="assets/vo.wav" data-start="{DIS:.2f}" data-track-index="9" data-volume="1"></audio>
    </div>
    <script>
      window.__timelines = window.__timelines || {{}};
      window.__timelines["main"] = gsap.timeline({{ paused: true }});
    </script>
  </body>
</html>
'''
(HERE / "index.html").write_text(html, encoding="utf-8")
import re
cf = HERE / "compositions" / "close.html"
cf.write_text(re.sub(r"const D = [\d.]+;", f"const D = {end_all - close0:.2f};", cf.read_text(encoding="utf-8")), encoding="utf-8")
print(f"voice ends {DIS + last['start'] + last['dur']:.1f} s; open {win['02-hook'][0]:.1f}-{win['02-hook'][1]:.1f}; close {close0:.1f}-{end_all:.1f} ({end_all - close0:.1f} s)")
