import numpy as np

from pipeline.search import (OVERSAMPLE, bls_numpy, central_duration, detrend, grid, search_star,
                             synthetic)


def test_detrend_flattens_variability_and_drops_a_flare():
    t, flux, sector = synthetic(depth=0.0)
    flux[500:503] += [0.03, 0.015, 0.007]
    lc = detrend(t, flux, sector)
    assert lc["n_flare"] >= 1
    assert not np.any((lc["t"] > t[500] - 0.001) & (lc["t"] < t[502] + 0.001))
    assert 800 < np.std(lc["y"]) * 1e6 < 1200  # back to the injected 1000 ppm white noise


def test_grid_keeps_phase_drift_within_a_fraction_of_the_duration():
    baseline = 2400.0
    freq, nbin, klo, khi = grid(baseline, 0.4, 0.4)
    drift = np.diff(freq) * baseline / freq[1:]  # days of phase drift across the baseline
    assert np.all(drift <= central_duration(1 / freq[1:], 0.4, 0.4) / OVERSAMPLE * 1.01)
    assert np.all(klo <= khi)


def test_injected_transit_passes_gate1():
    t, flux, sector = synthetic()
    row, peaks = search_star(t, flux, sector, 0.5, 0.5, bls_numpy, 1.0, 8.0)
    assert row["gate1"] == "pass", row["reason"]
    assert abs(row["dip_period"] - 3.217) < 0.01
    assert 1800 < row["dip_depth_ppm"] < 3200
    assert row["dip_transits"] >= 6


def test_noise_alone_fails_gate1():
    for seed in range(3):
        t, flux, sector = synthetic(seed=seed, depth=0.0)
        row, _ = search_star(t, flux, sector, 0.5, 0.5, bls_numpy, 1.0, 8.0)
        assert row["gate1"] == "fail"


def test_a_single_dip_is_not_a_periodic_signal():
    t, flux, sector = synthetic(depth=0.0)
    flux[np.abs(t - 2010.0) < 0.05] -= 0.01  # one deep event, SNR ~ 30 on its own
    row, peaks = search_star(t, flux, sector, 0.5, 0.5, bls_numpy, 1.0, 8.0)
    assert peaks[0]["side"] == "dip" and peaks[0]["share"] > 0.5  # the raw top peak is that one event
    assert row["gate1"] == "fail"


def test_calibrate_sets_the_floor_from_the_flipped_side(tmp_path):
    import csv
    from pipeline.search import RESULT_FIELDS, calibrate
    rng = np.random.default_rng(0)
    rows = [{"tic": n, "n_freq": 1000, "dip_snr": round(d, 2), "bump_snr": round(b, 2),
             "top_dip_snr": round(d, 2), "top_bump_snr": round(b, 2)}
            for n, (d, b) in enumerate(rng.uniform(5, 9, (300, 2)))]  # noise: dips and bumps alike
    rows += [{"tic": 999, "n_freq": 1000, "dip_snr": 20.0, "bump_snr": 7.0, "top_dip_snr": 20.0, "top_bump_snr": 7.0},
             {"tic": 998, "n_freq": 1000, "dip_snr": 12.0, "bump_snr": 30.0, "top_dip_snr": 12.0, "top_bump_snr": 30.0}]
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, RESULT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    summary = calibrate(path)
    assert 7.0 < summary["floor"] <= 9.0
    assert summary["flipped_pass"] <= 0.01 * len(rows)
    with path.open(newline="") as f:
        verdict = {int(r["tic"]): r["gate1"] for r in csv.DictReader(f)}
    assert verdict[999] == "pass" and verdict[998] == "fail"  # a variable star's bumps dwarf its dips
