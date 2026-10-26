import numpy as np
import pandas as pd

from pipeline.targets import expected_snr, noise_ppm_1hr


def test_noise_rises_with_magnitude_and_has_a_floor():
    noise = noise_ppm_1hr(np.array([4.0, 10.0, 13.5]))
    assert np.all(np.diff(noise) > 0)
    assert 59 < noise[0] < 70
    assert 100 < noise[1] < 300  # TESS 1-hour precision at Tmag 10 is a few hundred ppm


def test_same_planet_is_easier_to_see_around_a_smaller_star():
    stars = pd.DataFrame({
        "rad": [0.7, 0.3], "mass": [0.7, 0.3], "contratio": [0.0, 0.0],
        "tmag": [10.0, 10.0], "n_sectors": [4, 4],
    })
    snr = expected_snr(stars)
    assert snr[1] > snr[0]


def test_contamination_dilutes_the_signal():
    stars = pd.DataFrame({
        "rad": [0.6, 0.6], "mass": [0.6, 0.6], "contratio": [0.0, 0.5],
        "tmag": [11.0, 11.0], "n_sectors": [3, 3],
    })
    snr = expected_snr(stars)
    assert np.isclose(snr[1], snr[0] / 1.5)
