"""Cut the screen recordings into one mp4 per scene, each exactly as long as its window in the film.

Each scene maps source times to scene times with anchors [(src_s, scene_s), ...]; between two anchors the footage
plays at whatever constant speed makes it fit, so a moment (the verdict, the FPP line) lands on the word that names it.
Past the last anchor the last frame holds. The scene time "end" means the window's end.

  ../../.venv/Scripts/python cut.py            (build.py calls cut(), after the windows are known)
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "assets" / "footage"
OUT = SRC / "cut"
FF = Path("C:/Users/rushi/AppData/Local/Microsoft/WinGet/Packages/Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/"
          "ffmpeg-8.1.2-full_build/bin")

# run.webm (TIC 71503196 live, 10:33): search 0-74 s ("Cleared the bar" at 74), code checks from 85, Glimmer's cold
# start 91-184, its tools 184-205, the verdict at 229, gate 2 229-526, TRICERATOPS 527-626 (FPP at 626), done at 632.
PLAN = {
    "03-problem": ("03-problem.webm", None),
    "05-map": ("05-map.webm", [(0, 0), (7, 4.6), (30, "end")]),  # the hero numbers under "seventy-six thousand ... eight", then the map
    "06-search": ("run.webm", [(0, 0), (4, 2.4), (74, 10.0), (82, "end")]),
    "07-impostors": ("run.webm", [(82, 0), (126, "end")]),
    "08-glimmer": ("run.webm", [(178, 0), (228, 17.3), (242, "end")]),
    "09-triceratops": ("run.webm", [(522, 0), (625, 7.2), (633, 11.5)]),
    "10-accuracy": ("10-accuracy.webm", None),
    "11-join": ("11-join.webm", [(9, 0), (26, 5.0), (56, "end")]),  # the search under the URL line, then Ask Tessa
    "12-prediction": ("12-prediction.webm", None),
}


def length(f: Path) -> float:
    n = subprocess.run([str(FF / "ffprobe.exe"), "-v", "error", "-count_packets", "-select_streams", "v",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(f)],
                       capture_output=True, text=True, check=True).stdout.strip()
    return int(n) / 30  # every recording is 30 fps


def cut(windows: dict[str, tuple[float, float]]) -> dict[str, dict]:
    """Writes assets/footage/cut/<scene>.mp4 for each scene in PLAN; returns {scene: {src, speed}} (speed is the
    fastest stretch, for the label)."""
    OUT.mkdir(exist_ok=True)
    done = {}
    for scene, (name, anchors) in PLAN.items():
        if scene not in windows:
            continue
        s0, s1 = windows[scene]
        dur = s1 - s0
        f = SRC / name
        anchors = anchors or [(0, 0), (length(f), "end")]
        pts = [(a, dur if b == "end" else b) for a, b in anchors]
        parts, speeds = [], []
        for k, ((a0, b0), (a1, b1)) in enumerate(zip(pts, pts[1:])):
            factor = (b1 - b0) / (a1 - a0)  # scene seconds per source second
            speeds.append(1 / factor)
            parts.append(f"[0:v]trim={a0}:{a1},setpts=(PTS-STARTPTS)*{factor:.6f}[p{k}]")
        n = len(parts)
        graph = ";".join(parts) + ";" + "".join(f"[p{k}]" for k in range(n)) + \
            f"concat=n={n}:v=1:a=0,fps=30,tpad=stop_mode=clone:stop_duration={dur:.2f},trim=0:{dur:.3f}[v]"
        out = OUT / f"{scene}.mp4"
        sig = json.dumps([name, pts, round(dur, 3)])
        stamp = out.with_suffix(".json")
        if not (out.exists() and stamp.exists() and stamp.read_text() == sig):
            subprocess.run([str(FF / "ffmpeg.exe"), "-v", "error", "-y", "-i", str(f), "-filter_complex", graph,
                            "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
            stamp.write_text(sig)
        done[scene] = {"src": f"assets/footage/cut/{scene}.mp4", "speed": max(speeds)}
        print(f"{scene}: {dur:.1f} s from {name}, up to {max(speeds):.1f}x", flush=True)
    return done
