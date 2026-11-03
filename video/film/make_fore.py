"""The film's foreground silhouettes: a hill with grass and a big tree, drawn from seeded branching.

  ../../.venv/Scripts/python make_fore.py

Writes assets/tree.svg and assets/hill.svg, both 1920 x 1680 (the frame plus 600 px of canopy above it), in the
same coordinates, so the scene stacks them at one offset. The boy is drawn in the scene itself (he moves).
"""
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W, H = 1920, 1680
INK, INK2 = "#03050b", "#060913"
import sys
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 23)


def hill_y(x):
    # the ground line: a long rise from the left to the tree's knoll on the right
    return 1490 - 120 * math.exp(-((x - 1380) / 520) ** 2) + 30 * math.sin(x / 260) - 0.02 * (x - 900)


def hill():
    pts = [(x, hill_y(x)) for x in range(-20, W + 21, 10)]
    d = "M" + " L".join(f"{x:.0f},{y:.1f}" for x, y in pts) + f" L{W + 20},{H} L-20,{H} Z"
    blades = []
    for x in np.arange(-10, W + 10, 3.2):
        x += rng.uniform(-1.5, 1.5)
        base = hill_y(x) + 2
        h = rng.uniform(6, 22) * (1.4 if rng.random() < 0.08 else 1)
        lean = rng.uniform(-7, 7)
        blades.append(f"M{x - 1.6:.1f},{base:.1f} Q{x + lean * .4:.1f},{base - h * .6:.1f} {x + lean:.1f},{base - h:.1f} "
                      f"Q{x + lean * .3 + .8:.1f},{base - h * .5:.1f} {x + 1.6:.1f},{base:.1f}Z")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
            f'<path d="{d}" fill="{INK}"/><path d="{" ".join(blades)}" fill="{INK}"/></svg>')


def tree():
    """A broad old tree in the manner of a storybook ink drawing: a thick, twisting trunk that splits low into
    a few great limbs reaching wide, under a canopy wider than it is tall, its edge a fringe of single leaves."""
    limbs, clumps = [], []

    def grow(x, y, ang, length, width, depth):
        if depth == 0 or width < 2:
            for _ in range(3):
                clumps.append((x + rng.normal(0, 30), y + rng.normal(0, 20), rng.uniform(30, 52)))
            return
        bend = rng.normal(0, .22)
        mx = x + math.cos(ang + bend) * length * .5
        my = y - math.sin(ang + bend) * length * .5
        ex, ey = x + math.cos(ang) * length, y - math.sin(ang) * length
        w2 = width * .66
        limbs.append((x, y, mx, my, ex, ey, width, w2))
        if depth <= 4:
            for _ in range(2 if depth <= 2 else 1):
                clumps.append((ex + rng.normal(0, 40), ey + rng.normal(0, 26), rng.uniform(40, 72)))
        n = 2 if rng.random() < .7 else 3
        for k in range(n):
            a = ang + (k - (n - 1) / 2) * rng.uniform(.55, .85) + rng.normal(0, .1)
            a = min(max(a, .12), math.pi - .12)
            grow(ex, ey, a, length * rng.uniform(.68, .82), w2, depth - 1)

    bx, by = 1430, hill_y(1430) + 10
    # the trunk: one tapered, twisting path up to the fork
    fx, fy = bx - 30, by - 300
    trunk = (f"M{bx - 120},{by + 18} C{bx - 62},{by - 6} {bx - 52},{by - 60} {bx - 66},{by - 140} "
             f"C{bx - 82},{by - 220} {fx - 70},{fy + 30} {fx - 64},{fy} L{fx + 70},{fy} "
             f"C{fx + 60},{fy + 70} {bx + 30},{by - 170} {bx + 46},{by - 90} "
             f"C{bx + 54},{by - 40} {bx + 80},{by} {bx + 140},{by + 20} Z")
    # great limbs from the fork, reaching wide to both sides, the left one over the boy
    for ang, length, width in ((2.8, 250, 60), (2.25, 225, 62), (1.6, 200, 66), (1.0, 215, 60), (0.45, 230, 52)):
        grow(fx + rng.normal(0, 10), fy + 6, ang + rng.normal(0, .05), length, width, 6)

    def taper(x0, y0, mx, my, x1, y1, w0, w1):
        a = math.atan2(y1 - y0, x1 - x0) + math.pi / 2
        c, s = math.cos(a), math.sin(a)
        wm = (w0 + w1) / 2
        return (f"M{x0 + c * w0 / 2:.1f},{y0 + s * w0 / 2:.1f} Q{mx + c * wm / 2:.1f},{my + s * wm / 2:.1f} "
                f"{x1 + c * w1 / 2:.1f},{y1 + s * w1 / 2:.1f} L{x1 - c * w1 / 2:.1f},{y1 - s * w1 / 2:.1f} "
                f"Q{mx - c * wm / 2:.1f},{my - s * wm / 2:.1f} {x0 - c * w0 / 2:.1f},{y0 - s * w0 / 2:.1f}Z")
    limb_d = "".join(f'<path d="{taper(*l)}"/>' for l in limbs)  # one path each: overlaps never cancel
    # foliage: solid clumps, and around each a fringe of single leaves so the canopy's edge is lacy, not round
    body, fringe = [], []
    for x, y, r in clumps:
        body.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{r * .8:.0f}"/>')
        for _ in range(int(r * .55)):
            t = rng.uniform(0, 2 * math.pi)
            d = r * rng.uniform(.75, 1.08)
            lx, ly = x + math.cos(t) * d, y + math.sin(t) * d
            ang = math.degrees(t) + rng.normal(0, 25)
            fringe.append(f'<ellipse cx="{lx:.1f}" cy="{ly:.1f}" rx="{rng.uniform(7, 12):.1f}" ry="{rng.uniform(3, 5):.1f}" '
                          f'transform="rotate({ang:.0f} {lx:.1f} {ly:.1f})"/>')
    back = "".join(f'<circle cx="{x + rng.normal(0, 20):.0f}" cy="{y - 18:.0f}" r="{r:.0f}"/>' for x, y, r in clumps if rng.random() < .5)
    xs = [c[0] for c in clumps]; ys = [c[1] for c in clumps]
    print(f"canopy x {min(xs):.0f}..{max(xs):.0f}, y {min(ys):.0f}..{max(ys):.0f}")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">'
            f'<g fill="{INK2}">{back}</g><g fill="{INK}"><path d="{trunk}"/>{limb_d}{"".join(body)}{"".join(fringe)}</g></svg>')


(HERE / "assets/hill.svg").write_text(hill(), encoding="utf-8")
(HERE / "assets/tree.svg").write_text(tree(), encoding="utf-8")
print("trunk base", 1420, round(hill_y(1420)), "| boy sits at about x 1300, y", round(hill_y(1300)))
