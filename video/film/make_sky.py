"""The film's sky: the survey's 99,461 real stars (site/public/data/sky.json), drawn as two PNG layers for parallax.

  ../../.venv/Scripts/python make_sky.py

RA 95 to 370 degrees runs across the frame and Dec +50 to -62 down a 3000 px strip (stretched vertically so the
camera can pan up it). Brighter stars (by TESS magnitude) go on the near layer, the rest on the far one. Writes
assets/sky-far.png, assets/sky-near.png and assets/cands.json (the 7 candidates' pixel positions, for the close).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
W, H, SS = 1920, 3000, 2           # strip size, supersampling
RA0, RA1, DEC0, DEC1 = 95.0, 370.0, 50.0, -62.0

d = json.loads((HERE.parent.parent / "site/public/data/sky.json").read_text(encoding="utf-8"))
ra = np.array(d["ra"]); ra = np.where(ra < RA0, ra + 360, ra)
dec, tmag, teff = np.array(d["dec"]), np.array(d["tmag"]), np.array(d["teff"], float)
x = (ra - RA0) / (RA1 - RA0) * W
y = (DEC0 - dec) / (DEC0 - DEC1) * H
inside = (x > 0) & (x < W) & (y > 0) & (y < H)


def fill_gaps():
    """TESS skipped some sky (near the ecliptic, early on), which leaves holes no night sky has. Fill each hole up
    to the strip's usual density with stars drawn from the survey's own magnitudes and temperatures (seeded)."""
    global x, y, tmag, teff, ra, dec, inside
    C = 60
    hist, _, _ = np.histogram2d(y[inside], x[inside], bins=(H // C, W // C), range=((0, H), (0, W)))
    pad = np.pad(hist, 1, mode="edge")
    hb = sum(pad[a:a + hist.shape[0], b:b + hist.shape[1]] for a in range(3) for b in range(3)) / 9
    want = np.median(hist)
    nr = np.random.default_rng(5)
    add_x, add_y, pool = [], [], np.flatnonzero(inside)
    for (i, j), have in np.ndenumerate(hist):
        k = int(max(0, want - max(have, 0.6 * hb[i, j]) if have < 0.6 * want else 0))
        add_x += list(nr.uniform(j * C, (j + 1) * C, k)); add_y += list(nr.uniform(i * C, (i + 1) * C, k))
    src = nr.choice(pool, len(add_x))
    x = np.concatenate([x, add_x]); y = np.concatenate([y, add_y])
    tmag = np.concatenate([tmag, tmag[src]]); teff = np.concatenate([teff, teff[src]])
    inside = np.concatenate([inside, np.ones(len(add_x), bool)])
    print(len(add_x), "stars filled into gaps")


fill_gaps()


def tint(t):
    """Star colour from temperature, pulled most of the way to white: red dwarfs read warm, not orange."""
    t = np.clip(t, 2500, 7000)
    r = np.ones_like(t)
    g = np.clip(0.39 * np.log(t / 100) - 0.63, 0, 1)
    b = np.clip(0.54 * np.log(t / 100 - 10) - 1.19, 0, 1)
    c = np.stack([r, g, b], 1)
    return 0.35 * c + 0.65 * np.array([0.86, 0.9, 1.0])


rng = np.random.default_rng(7)  # only for sub-pixel placement jitter, fixed
flux = 10 ** (-0.4 * (tmag - 9.0))
col = tint(teff)


def layer(mask, name, gain):
    img = np.zeros((H * SS, W * SS, 3), np.float32)
    for i in np.flatnonzero(mask):
        f = flux[i]
        sig = SS * (0.45 + 0.9 * min(f, 2.5) ** 0.5)
        amp = gain * min(0.08 + 0.9 * f ** 0.8, 1.6)
        cx, cy = x[i] * SS + rng.uniform(-.5, .5), y[i] * SS + rng.uniform(-.5, .5)
        r = int(np.ceil(sig * 3.2))
        x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, W * SS)
        y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, H * SS)
        gx = np.exp(-((np.arange(x0, x1) - cx) ** 2) / (2 * sig ** 2))
        gy = np.exp(-((np.arange(y0, y1) - cy) ** 2) / (2 * sig ** 2))
        img[y0:y1, x0:x1] += amp * np.outer(gy, gx)[..., None] * col[i]
    img = img.reshape(H, SS, W, SS, 3).mean((1, 3))
    a = np.clip(img.max(2), 0, 1)
    rgb = np.where(a[..., None] > 0, img / np.maximum(img.max(2, keepdims=True), 1e-6), 0)
    out = np.dstack([np.clip(rgb, 0, 1) * 255, a * 255]).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(HERE / "assets" / name, optimize=True)
    print(name, int(mask.sum()), "stars")


def milky_way():
    """The survey skips the crowded galactic plane, so its stars leave a band empty: that band is the Milky Way.
    Drawn as a soft glow with dust lanes and a haze of faint stars, from seeded noise, where |b| is small."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.radians(RA0 + xx / W * (RA1 - RA0)); dd = np.radians(DEC0 - yy / H * (DEC0 - DEC1))
    ag, dg = np.radians(192.85948), np.radians(27.12825)
    sb = np.sin(dd) * np.sin(dg) + np.cos(dd) * np.cos(dg) * np.cos(r - ag)
    b = np.degrees(np.arcsin(np.clip(sb, -1, 1)))
    nr = np.random.default_rng(11)

    def noise(cells):
        out = np.zeros((H, W), np.float32)
        for k, amp in ((cells, 1), (cells * 2, .5), (cells * 4, .25), (cells * 8, .12)):
            g = nr.random((int(H / W * k) + 2, k + 2)).astype(np.float32)
            im = Image.fromarray((g * 255).astype(np.uint8)).resize((W, H), Image.BICUBIC)
            out += amp * np.asarray(im, np.float32) / 255
        return out / 1.87
    band = np.exp(-(b / 7) ** 2)
    glow = band * (0.15 + 1.1 * noise(4) ** 3)
    dust = np.exp(-((b + 1.5 * (noise(5) - .5) * 6) / 2.2) ** 2) * noise(9)
    glow = np.clip(glow * (1 - 0.75 * dust), 0, None)
    img = glow[..., None] * np.array([0.42, 0.45, 0.6], np.float32) * 0.55
    n = 26000  # the haze of stars too faint and crowded for the survey
    ys = nr.integers(0, H, n * 6); xs = nr.integers(0, W, n * 6)
    keep = nr.random(n * 6) < glow[ys, xs] / glow.max()
    ys, xs = ys[keep][:n], xs[keep][:n]
    br = nr.random(len(ys)) ** 3 * 0.8 + 0.12
    img[ys, xs] += br[:, None] * np.array([0.9, 0.92, 1.0])
    a = np.clip(img.max(2), 0, 1)
    rgb = img / np.maximum(img.max(2, keepdims=True), 1e-6)
    out = np.dstack([np.clip(rgb, 0, 1) * 255, a * 255]).astype(np.uint8)
    Image.fromarray(out, "RGBA").save(HERE / "assets" / "sky-milky.png", optimize=True)
    print("sky-milky.png")


milky_way()
near = inside & (tmag < 11.2)
layer(inside & ~near, "sky-far.png", 0.55)
layer(near, "sky-near.png", 1.0)
cands = [{"tic": d["tic"][i], "x": round(float(x[i]), 1), "y": round(float(y[i]), 1)}
         for i in range(len(d["tic"])) if d["status"][i] == d["codes"]["candidate"] and inside[i]]
(HERE / "assets" / "cands.json").write_text(json.dumps(cands, indent=1), encoding="utf-8")
print(len(cands), "candidates in the strip")
