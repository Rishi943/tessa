from pipeline.vet import RULE_REJECT_RANK, junk, verdict


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


VARIES = {"ratio_to_depth": 0.9, "chi2": 300.0, "amplitude_ppm": 900.0}
ON_GAIA = {"verdict": "off_target", "on_star": {"source_id": 7, "sep_arcsec": 20.0}, "offset_sigma": 8.0, "offset_arcsec": 20.0}


def test_the_star_varying_and_a_dip_on_a_gaia_star_reject_alone_only_at_a_low_rank():
    ok = {"p_planet": 0.6, "cut": 0.12, "passes": True, "planet_rank": 0.5, "drivers": []}
    for over in ({"phase_curve": VARIES}, {"centroid": ON_GAIA}):
        assert verdict(checks(odd_even={"diff_sigma": 1.0}, system1=ok, **over)) == ("agent", [])  # rank 0.5: a flag for the agent
        low = dict(ok, planet_rank=RULE_REJECT_RANK - 0.01)
        assert verdict(checks(odd_even={"diff_sigma": 1.0}, system1=low, **over))[0] == "reject"
        assert verdict(checks(odd_even={"diff_sigma": 1.0}, **over))[0] == "reject"  # no System 1 score: as before v9
        v, why = verdict(checks(odd_even={"diff_sigma": 1.0}, system1=dict(ok, passes=False), **over))  # another reason: both listed
        assert v == "reject" and len(why) == 2


def test_a_varying_star_still_gets_a_system1_score():
    assert junk(checks(phase_curve=VARIES)) == []  # junk skips System 1, the centroid and the tools
    assert junk(checks(duration={"ratio": 4.0}))
