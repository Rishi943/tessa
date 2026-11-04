"""Arm 1: a hand-written scorer. No optimizer, no fitted parameter: every number below was written by the
model after reading train-split summaries (explore.py, bins.py) and the physics of each check.

Form: an evidence ledger. Each clue adds log-odds for "planet" through a curve given as (value, log-odds)
points, interpolated linearly (in log10 of the value where the clue spans decades) and held flat past the
ends. Clues that measure the same thing (V-shape, ingress, impact parameter) are down-weighted so the
same evidence is not counted twice. Margin = sum of weighted terms; calibration happens in evaluate.py.

Revision log (each checked against train only):
  v1: first draft (train AUC 0.946, FPs caught at 2% planet loss 0.704)
  v2: host star judged with the dip's size; RV error read with temperature (0.947, 0.727)
  v3: full weight for clues that still split the survivors: Gaia RV, nearest contaminant, density penalty,
      contamination, half-phase dip. FROZEN after v3.
"""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent / "work"
NAMES = json.loads((WORK / "features.json").read_text(encoding="utf-8"))

# (feature, weight, log-axis?, value if missing, curve points)
LEDGER = [
    # --- shape: a planet's transit is U-shaped (flat bottom); grazing or blended EBs are V-shaped ---
    ("v_shape_sigma", 1.0, False, 0.0, [(-40, 1.6), (-15, 1.4), (-7, 1.0), (-3, 0.6), (-1.5, 0.1), (0, -0.6),
                                        (1.5, -1.6), (4, -2.3), (10, -2.2)]),
    ("ingress_fraction", 0.5, False, 0.0, [(0.1, 0.2), (0.15, 1.3), (0.25, 1.3), (0.3, 0.7), (0.35, -0.3),
                                           (0.4, -1.2), (0.45, -2.2), (0.5, -2.4)]),
    ("fit_b", 0.5, False, 0.0, [(0, 1.4), (0.2, 0.6), (0.5, 0.7), (0.7, 0.5), (0.85, -0.3), (0.9, -0.8),
                                (0.94, -1.7), (0.98, -2.6)]),
    # --- star density: the transit's duration must fit the catalogue star; a mismatch means a different star ---
    ("duration_ratio", 0.5, False, 0.0, [(0.3, -2.0), (0.45, -1.0), (0.6, -0.5), (0.7, 0.4), (0.85, 1.2),
                                         (1.0, 1.1), (1.1, 0.3), (1.3, -1.4), (2.0, -2.0)]),
    ("fit_log_density_ratio", 0.4, False, 0.0, [(-1, -1.9), (-0.5, -0.9), (-0.2, -0.4), (0, 1.1), (0.15, 1.3),
                                                (0.3, 0.4), (0.6, 0.1), (1, -1.1)]),
    ("star_density_penalty_share", 0.8, True, -2.0, [(1e-6, -0.2), (1e-5, 1.6), (3e-4, 1.3), (1e-3, 0.7),
                                                     (2.3e-3, 0.3), (6e-3, -0.1), (0.015, -1.3), (0.04, -1.8), (0.1, -2.1)]),
    # --- blending: is the dip on the target, or on a neighbour? ---
    ("centroid_offset_sigma", 1.0, True, -0.35, [(0.3, 2.0), (0.9, 1.4), (1.3, 1.3), (2, 0.65), (3.3, 0.1),
                                                 (6, -0.9), (10, -1.3), (18, -2.0)]),
    ("centroid_offset_arcsec", 0.3, True, 0.0, [(0.3, 1.7), (1.3, 1.2), (2.5, 1.1), (4, 0.6), (7, 0.0),
                                                (13, -0.85), (21, -1.45), (31, -2.3)]),
    ("closest_could_sep", 0.9, True, 1.1, [(3, -0.8), (8, -1.2), (12, -0.75), (16, -0.6), (23, 0.0), (32, 0.45),
                                           (45, 0.8), (70, 1.4), (150, 2.1)]),
    ("plausible_within_1px", 0.5, False, 0.0, [(1, 0.9), (2, -0.5), (3, -1.1), (4, -1.7)]),
    ("contamination", 0.6, True, -0.6, [(0.002, 2.0), (0.006, 1.4), (0.012, 0.7), (0.023, 0.6), (0.04, 0.1),
                                        (0.07, -0.3), (0.12, -0.6), (0.2, -0.9)]),
    # a neighbour that Gaia lists as an eclipsing binary at this period is the answer itself
    ("gaia_eb_neighbour_match", 1.0, True, 0.0, [(0.02, -3.0), (0.1, -2.5), (0.6, -2.0), (2, -1.3)]),
    # --- eclipsing binary: a second eclipse, a dip at half phase ---
    ("secondary_over_bump", 0.6, False, 0.0, [(-0.35, 2.0), (0.08, 0.85), (0.4, -0.1), (1.25, -0.05),
                                              (1.6, -0.7), (3, -1.4)]),
    ("secondary_snr", 0.4, False, 0.0, [(-3, 1.8), (0.4, 1.0), (1.5, -0.1), (3.3, 0.3), (4.3, -0.35),
                                        (7.8, -1.15), (15, -1.7)]),
    ("sector_sec_half_max_snr", 0.6, False, 0.0, [(-1.5, 1.7), (-0.2, 0.85), (0.35, 0.65), (1.1, 0.4),
                                                  (1.5, -0.15), (2.6, -0.5), (3.9, -1.15), (8, -1.8)]),
    ("secondary_phase_offset", 0.5, False, 0.0, [(0.0036, -1.45), (0.013, -0.75), (0.04, 0.0), (0.11, 0.3)]),
    # --- the star: big or hot stars host more false positives; Gaia binarity signs ---
    # (star_radius, teff, planet_radius: v2 moved them to host_and_size below)
    ("distance", 0.3, True, 0.0, [(60, 1.7), (110, 1.0), (160, 0.35), (210, 0.0), (400, -0.15), (550, -0.7),
                                  (800, -0.7), (1500, -1.0)]),
    ("ruwe", 0.7, False, -1.5, [(1.2, 0.15), (1.25, -0.25), (1.6, -0.3), (2.0, -1.65)]),
    ("rv_amplitude", 1.0, True, 0.0, [(2, 0.8), (12, 0.45), (18, -0.1), (36, -1.3), (80, -2.9)]),
    ("rv_gof", 0.5, False, 0.0, [(0, 0.3), (3, 0.0), (10, -0.4), (20, -2.5)]),
    ("excess_noise_sig", 0.3, True, 0.0, [(10, 0.3), (30, -0.25), (92, -0.4), (1000, -1.3)]),
    ("non_single_star", 1.0, False, 0.0, [(0, 0.0), (1, -1.0)]),
    # --- the signal: very short periods and Neptune-desert sizes are mostly binaries ---
    ("period", 0.6, True, 0.0, [(1, -1.2), (1.6, -0.8), (2.2, -0.35), (2.8, -0.25), (3.4, 0.3), (4.5, 0.7),
                                (6.5, 0.3), (10, 0.45), (15, 0.2)]),
    ("planet_radius", 1.0, True, 0.0, [(22, 0.0), (30, -1.0), (50, -2.0)]),  # bigger than any planet
    ("snr", 0.4, True, 0.0, [(12, -0.25), (30, -0.45), (47, -0.2), (73, -0.1), (123, 0.3), (250, 0.9), (500, 1.15)]),
    ("transits", 0.3, True, 0.0, [(10, 0.6), (27, 0.85), (35, 0.1), (46, -0.15), (66, -0.45), (164, -0.9)]),
    # --- light-curve quality: a dip small next to the star's own swings is often the star ---
    ("variability_over_depth", 0.8, True, 0.0, [(0.02, 0.9), (0.05, 1.2), (0.08, 0.55), (0.11, 0.3), (0.15, 0.2),
                                                (0.2, -0.25), (0.27, -0.75), (0.4, -1.45), (1, -2.6)]),
    ("red_noise", 0.4, False, 0.0, [(1.02, 0.4), (1.05, -0.1), (1.16, 0.0), (1.27, -0.55), (1.5, -0.85)]),
    ("odd_even_fraction", 0.5, True, 0.0, [(0.008, 0.65), (0.02, 0.6), (0.03, 0.3), (0.05, 0.0), (0.1, -0.1),
                                           (0.16, -0.25), (0.24, -0.4), (0.5, -0.9)]),
    ("gap_p_chance", 0.4, False, 0.0, [(0.1, -0.6), (0.27, -0.2), (0.4, 0.1), (0.8, 0.3), (1, 0.65)]),
]


def term(col: np.ndarray, log: bool, missing: float, pts) -> np.ndarray:
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    ok = np.isfinite(col)
    v = col.copy()
    if log:
        xs = np.log10(xs)
        v = np.where(ok & (v > 0), np.log10(np.where(v > 0, v, 1)), xs[0])  # zero or negative: the low end
    out = np.full(len(col), missing, float)
    out[ok] = np.interp(v[ok], xs, ys)
    return out


def curve(v, pts, log=False):
    return term(np.asarray(v, float), log, 0.0, pts)


# The host star judged together with the dip's size (v2). Hot Jupiters around F stars are common and real;
# a small dip on a hot or large star is almost always a blend, a pulsation or a binary. One curve per size
# (anchored at 2.5, 6.5 and 14 Earth radii), blended in log radius between anchors.
TEFF_BY_SIZE = {
    2.5: [(3800, 1.5), (4500, 0.8), (5250, 0.65), (5750, 0.55), (6100, -0.35), (6350, -0.3), (6650, -2.5), (8000, -3.0)],
    6.5: [(3800, 0.0), (4500, 0.15), (5250, -0.35), (5750, -0.5), (6100, -1.9), (6350, -1.6), (6650, -2.7), (7000, -3.0)],
    14: [(3800, 0.2), (4500, 0.4), (5250, 1.0), (5750, 1.05), (6100, 0.6), (6350, 0.25), (6650, 0.0), (7000, -1.25),
         (8000, -1.15)],
}
RSTAR_BY_SIZE = {
    2.5: [(0.5, 1.0), (0.75, 0.75), (1.05, 0.0), (1.35, -0.6), (1.75, -1.7), (2.5, -2.7), (4, -2.1)],
    6.5: [(0.5, 1.0), (0.75, 0.75), (1.05, 0.0), (1.35, -0.6), (1.75, -1.7), (2.5, -2.7), (4, -2.1)],
    14: [(0.5, 0.45), (0.75, 1.45), (1.05, 1.1), (1.35, 0.7), (1.75, 0.35), (2.5, -0.7), (4, -2.6)],
}


def by_size(value, rp, table, log=False):
    anchors = sorted(table)
    cols = np.stack([curve(value, table[a], log) for a in anchors], axis=1)
    lr = np.log10(np.clip(np.nan_to_num(rp, nan=6.5), 0.1, None))
    la = np.log10(anchors)
    out = np.empty(len(lr))
    for i, x in enumerate(lr):
        out[i] = np.interp(x, la, cols[i])
    return np.where(np.isfinite(value), out, 0.0)


def rv_error_term(rv_err, teff):
    """Gaia's RV error grows when the star's lines shift between epochs (a binary); hot stars' broad lines make
    it grow anyway, but a hot star with a large error is still mostly binaries in train."""
    cool = curve(rv_err, [(0.3, 0.7), (0.5, 0.9), (1, 0.8), (1.5, 0.35), (3, -0.45)], log=True)
    hot = curve(rv_err, [(0.3, -0.1), (0.5, 0.3), (0.75, -0.75), (1.5, -1.4), (3, -2.6)], log=True)
    out = np.where(np.nan_to_num(teff, nan=5800) >= 6200, hot, cool)
    return np.where(np.isfinite(rv_err), out, 0.0)


def get(X, n):
    return X[:, NAMES.index(n)]


SPECIAL = [
    ("host teff x size", 0.6, lambda X: by_size(get(X, "teff"), get(X, "planet_radius"), TEFF_BY_SIZE)),
    ("host radius x size", 0.5, lambda X: by_size(get(X, "star_radius"), get(X, "planet_radius"), RSTAR_BY_SIZE, log=True)),
    ("Gaia RV error x teff", 0.8, lambda X: rv_error_term(get(X, "rv_error"), get(X, "teff"))),
]
TERM_NAMES = [e[0] for e in LEDGER] + [s[0] for s in SPECIAL]


def contributions(X: np.ndarray) -> np.ndarray:
    cols = [w * term(X[:, NAMES.index(n)], log, miss, pts) for n, w, log, miss, pts in LEDGER]
    cols += [w * fn(X) for _, w, fn in SPECIAL]
    return np.stack(cols, axis=1)


def margin(X: np.ndarray) -> np.ndarray:
    return contributions(X).sum(axis=1)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "hand"
    out = {s: margin(np.load(WORK / f"{s}.npz")["X"]) for s in ("train", "val", "test")}
    np.savez(WORK / f"margins-{name}.npz", **out)
    print(f"wrote margins-{name}.npz")


if __name__ == "__main__":
    main()
