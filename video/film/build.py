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
ap.add_argument("--vo", default="me-telescope-42")  # the cloned voice (drama-yt-42: the British AI take)
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
# subtitles: every spoken line, cut into short pieces at word boundaries (at most CAP characters, breaking after
# punctuation where it can) and timed across the line by length; spoken forms written as a reader expects them
CAP = 44
WRITTEN = [("X G Boost", "XGBoost"), ("tessa dot brave A I dot solutions", "tessa.braveai.solutions"),
           ("ExoMiner plus plus", "ExoMiner++"), ("seventy-six thousand", "76,000"),
           ("seven hundred and forty-five", "745"), ("five hundred and seventy-nine", "579"),
           ("ninety-two percent", "92%"), ("two hundred thousand", "200,000"), ("three hundred", "300"),
           ("two hundred light-years", "200 light-years"), ("under one percent", "under 1%")]


def pieces(text):
    for a_, b_ in WRITTEN:
        text = text.replace(a_, b_)
    # sentences first, then clauses at commas and colons packed up to CAP, then a long clause cut into even parts
    import math as _m, re as _re
    out = []
    for sent in _re.split(r"(?<=[.?!])\s+", text):
        cur = ""
        for cl in _re.split(r"(?<=[,:])\s+", sent):
            if cur and len(cur) + 1 + len(cl) <= CAP:
                cur = f"{cur} {cl}"
                continue
            if cur:
                out.append(cur)
            if len(cl) <= CAP:
                cur = cl
                continue
            w = cl.split(); n = _m.ceil(len(cl) / CAP); k = _m.ceil(len(w) / n)
            parts = [" ".join(w[j:j + k]) for j in range(0, len(w), k)]
            out += parts[:-1]; cur = parts[-1]
        if cur:
            out.append(cur)
    k = 0  # a stranded scrap ("Here,", "anything.") joins its neighbour when the two fit on one line
    while k < len(out):
        if len(out[k]) < 16 and len(out) > 1:
            j = k + 1 if k + 1 < len(out) and out[k][-1] not in ".?!" else k - 1
            if 0 <= j < len(out) and len(out[k]) + len(out[j]) + 1 <= CAP + 10:
                a_, b_ = sorted((k, j))
                out[a_:b_ + 1] = [f"{out[a_]} {out[b_]}"]
                continue
        k += 1
    return out


caps = []
for i in ids:
    for ln in sc[i]["lines"]:
        ps = pieces(ln["text"])
        tot = sum(len(p) for p in ps)
        t0 = DIS + ln["start"]
        for p in ps:
            d = ln["dur"] * len(p) / tot
            caps.append(f'      <div id="cap-{len(caps)}" class="clip cap" data-start="{t0:.2f}" data-duration="{d:.2f}" '
                        f'data-track-index="4"><span>{p}</span></div>')
            t0 += d

# music: Debussy's Clair de lune (Laurens Goedhart, 2011, public domain, Wikimedia Commons), levelled, light under
# the voice, up to full once the voice ends so its second swell carries the close, faded out at the end
import subprocess
from cut import FF
MUSIC_SRC = HERE / "assets" / "music-src" / "clair-de-lune.ogg"  # from commons.wikimedia.org/wiki/File:Clair_de_lune_(Claude_Debussy)_Suite_bergamasque.ogg
MUSIC_IN = 2.0           # seconds into the recording at the film's start: its swell at about 2:50 lands on the close
UNDER, FULL = 0.10, 0.5  # about 12 dB under the voice while it speaks; on the close, a little above where it sat
v_end = DIS + last["start"] + last["dur"]
vol = (f"if(lt(t,{v_end - 0.3:.2f}),{UNDER},if(lt(t,{v_end + 1.2:.2f}),{UNDER}+({FULL}-{UNDER})*(t-{v_end - 0.3:.2f})/1.5,{FULL}))")
af = (f"atrim=start={MUSIC_IN},asetpts=PTS-STARTPTS,loudnorm=I=-20:TP=-2:LRA=11,volume='{vol}':eval=frame,"
      f"afade=t=in:d=2,afade=t=out:st={end_all - 1.6:.2f}:d=1.6,atrim=end={end_all:.2f}")
music = HERE / "assets" / "music.wav"
sig = music.with_suffix(".json")
if not (music.exists() and sig.exists() and sig.read_text() == af):
    subprocess.run([str(FF / "ffmpeg.exe"), "-v", "error", "-y", "-i", str(MUSIC_SRC), "-af", af, "-ar", "48000", "-ac", "2", str(music)], check=True)
    sig.write_text(af)

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
      .cap {{ position: absolute; left: 0; right: 0; bottom: 118px; z-index: 6; text-align: center; }}
      .cap span {{ display: inline-block; max-width: 1400px; padding: 10px 22px; border-radius: 8px;
                   background: rgba(6, 8, 14, 0.72); color: #F4F3EE; font: 500 40px/1.25 Archivo, sans-serif; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{end_all:.2f}" data-width="1920" data-height="1080">
{chr(10).join(hosts)}
{chr(10).join(holders)}
{chr(10).join(lab)}
{chr(10).join(caps)}
      <audio id="vo" class="clip" src="assets/vo.wav" data-start="{DIS:.2f}" data-track-index="9" data-volume="1"></audio>
      <audio id="music" class="clip" src="assets/music.wav" data-start="0" data-duration="{end_all:.2f}" data-track-index="10" data-volume="1"></audio>
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
