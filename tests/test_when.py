import numpy as np

from pipeline.gate2 import outside_planet_range
from pipeline.search import detrend
from pipeline.vet import phase_offset
from pipeline.when import ephemeris, predict

PERIOD, T0, DUR = 3.217, 2001.3, 0.08


def two_sectors(seed=3):
    """Box transits in two sectors 400 days apart, 10-minute cadence, 1000 ppm noise."""
    rng = np.random.default_rng(seed)
    t = np.concatenate([np.arange(2000, 2027, 1 / 144), np.arange(2400, 2427, 1 / 144)])
    flux = 1 + rng.normal(0, 1000e-6, len(t))
    flux[np.abs(phase_offset(t, PERIOD, T0)) < DUR / 2] -= 2500e-6
    return t, flux, np.where(t < 2200, 5, 20).astype(np.int16)


def test_ephemeris_corrects_a_slightly_wrong_search_period():
    lc = detrend(*two_sectors())
    sig = {"period": PERIOD + 2e-5, "t0": T0 + 0.002, "dur": DUR, "depth_ppm": 2500.0, "snr": 30.0}
    eph = ephemeris(lc, sig)
    assert eph["ok"] and eph["timed_per"] == "sector" and eph["groups"] == 2
    assert eph["period_err"] < 2e-5
    assert abs(eph["period"] - PERIOD) < 4 * eph["period_err"]
    mid, err = predict(eph, np.array([T0 + 500 * PERIOD]))
    assert abs(mid[0] - (T0 + 500 * PERIOD)) < 4 * err[0]


def test_timing_error_grows_with_distance_from_the_data():
    lc = detrend(*two_sectors())
    eph = ephemeris(lc, {"period": PERIOD, "t0": T0, "dur": DUR, "depth_ppm": 2500.0, "snr": 30.0})
    _, err = predict(eph, np.array([2410.0, 2410.0 + 100 * PERIOD, 2410.0 + 1000 * PERIOD]))
    assert err[0] < err[1] < err[2]


def test_gate2_rules_follow_the_agents_table():
    checks = {"odd_even": {"diff_sigma": 0.5}, "secondary": {"snr": 6.0, "phase": 0.5, "best_brightening_snr": 4.5},
              "sectors": {"p_constant": 0.4}, "size": {"rp_rearth": 2.2}, "duration": {"ratio": 0.8}, "known": [],
              "shape": {"v_sigma": -3.0}, "neighbours": {"target_found": True, "plausible_within_1px": 1},
              "centroid": {"verdict": "on_target"}, "phase_curve": {"ratio_to_depth": 0.2, "chi2": 80.0},
              "refit": {"transits": 40}, "signal": {"period": 2.5}}
    assert outside_planet_range(checks) == []  # the blend is cleared by the centroid
    checks["centroid"]["verdict"] = "inconclusive"
    assert [r.split(":")[0] for r in outside_planet_range(checks)] == ["neighbours"]
    checks["centroid"]["verdict"] = "on_target"
    checks["secondary"]["best_brightening_snr"] = 3.0  # 6 is now above 1.5 x the noise
    assert [r.split(":")[0] for r in outside_planet_range(checks)] == ["secondary"]
