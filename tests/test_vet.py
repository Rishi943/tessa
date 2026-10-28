from pipeline.vet import verdict


def checks(**over):
    c = {"tic": 1, "known": [], "secondary": {"snr": 2.0, "phase": 0.5, "best_brightening_snr": 2.0},
         "odd_even": {"diff_sigma": 5.0}, "size": {"rp_rearth": 2.0},
         "phase_curve": {"ratio_to_depth": 0.1, "chi2": 10.0, "amplitude_ppm": 50.0}, "duration": {"ratio": 1.0}}
    c.update(over)
    return c


def test_odd_even_is_a_veto_only_without_system1():
    assert verdict(checks())[0] == "reject"
    passes = {"p_planet": 0.6, "cut": 0.12, "passes": True, "planet_rank": 0.2, "drivers": []}
    assert verdict(checks(system1=passes)) == ("agent", [])
    fails = dict(passes, p_planet=0.05, passes=False, planet_rank=0.0,
                 drivers=[{"label": "odd/even difference (sigma)", "value": 5.0, "log_odds": -1.2}])
    v, why = verdict(checks(system1=fails))
    assert v == "reject" and why[0].startswith("System 1: P(planet) 0.050") and "odd/even" in why[0]


def test_junk_stays_a_veto_with_system1():
    passes = {"p_planet": 0.6, "cut": 0.12, "passes": True, "planet_rank": 0.2, "drivers": []}
    v, why = verdict(checks(system1=passes, duration={"ratio": 4.0}))
    assert v == "reject" and why[0].startswith("transit 4.0x longer")


def test_unmeasured_checks_do_not_reject():
    c = checks(odd_even={"diff_sigma": None}, secondary={"snr": None, "phase": None, "best_brightening_snr": None},
               size={"rp_rearth": None}, phase_curve={"ratio_to_depth": None, "chi2": None, "amplitude_ppm": None},
               duration={"ratio": None})
    assert verdict(c) == ("agent", [])
