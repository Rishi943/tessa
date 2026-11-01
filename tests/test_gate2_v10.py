"""Vetting v10's gate 2 rule, TRICERATOPS's classes and the invented-number guard on dossiers."""
import json

from finetune.pilot import invented, parse
from pipeline.fpp import classify
from pipeline.gate2 import FLOOR_RANK, PASS_RANK, outside_planet_range


def planet_like(rank: float) -> dict:
    return {"odd_even": {"diff_sigma": 0.5}, "secondary": {"snr": 1.0, "phase": 0.5, "best_brightening_snr": 3.0},
            "sectors": {"p_constant": 0.4}, "size": {"rp_rearth": 2.2}, "duration": {"ratio": 0.8}, "known": [],
            "shape": {"v_sigma": 0.0}, "neighbours": {"target_found": True, "plausible_within_1px": 0},
            "centroid": {"verdict": "on_target"}, "phase_curve": {"ratio_to_depth": 0.2, "chi2": 80.0},
            "refit": {"transits": 40}, "signal": {"period": 2.5}, "system1": {"planet_rank": rank}}


def test_v10_lets_the_agent_pass_below_the_v9_cut():
    checks = planet_like(PASS_RANK - 0.2)
    assert [r.split(":")[0] for r in outside_planet_range(checks)] == ["system1"]  # v9: the rank alone stops it
    assert outside_planet_range(checks, version="v10") == []  # v10: the agent's call stands


def test_v10_drops_the_soft_rows_but_keeps_the_hard_ones():
    checks = planet_like(0.1)
    checks["secondary"]["snr"] = 6.0  # a table row the v9 rule would flag
    assert outside_planet_range(checks, version="v10") == []
    checks["neighbours"]["target_found"] = False  # a HARD_ROWS row: the blend check could not be made
    assert [r.split(":")[0] for r in outside_planet_range(checks, version="v10")] == ["neighbours"]
    checks["neighbours"]["target_found"] = True
    checks["known"] = [{"harmonic": 1.0}]
    assert [r.split(":")[0] for r in outside_planet_range(checks, version="v10")] == ["known"]


def test_v10_floor_caps_a_pass_the_rank_puts_at_the_bottom():
    assert outside_planet_range(planet_like(FLOOR_RANK), version="v10") == []
    why = outside_planet_range(planet_like(FLOOR_RANK - 0.01), version="v10")
    assert [r.split(":")[0] for r in why] == ["system1"] and "human" in why[0]


def test_triceratops_classes_follow_giacalone_2021():
    assert classify(0.009, 0.0004) == "validated"
    assert classify(0.06, 0.0004) == "likely planet"
    assert classify(0.6, 0.0004) == "unclassified"
    assert classify(0.05, 0.002) == "unclassified"  # the survey keeps these as "nearby star to rule out"
    assert classify(0.15, 0.12) == "likely nearby false positive"
    assert classify(None, 0.001) == "unclassified"


def dossier(value, why: str) -> str:
    return "tool call first\n" + json.dumps({"verdict": "gate2", "reasons": [{"check": "depth", "value": value, "why": why}]})


def test_invented_numbers_are_caught_and_inputs_are_not():
    rec = {"user": "depth 1840 ppm, period 2.4371 d, rank 0.47", "answers": {"rotation": {"rotation_d": 11.3}}}
    assert invented(rec | {"text": dossier(1840, "depth 1840 ppm over 2.4371 d")}, "") == []
    assert invented(rec | {"text": dossier(1850, "close to the input, within 3%")}, "") == []
    assert invented(rec | {"text": dossier("", "spots turn every 11.3 d")}, "") == []  # tool answers count as inputs
    assert invented(rec | {"text": dossier("", "rank 47%")}, "") == []  # a fraction written as a percentage
    assert invented(rec | {"text": dossier("", "3 transits, odd and even agree")}, "") == []  # small counts are allowed
    assert invented(rec | {"text": dossier(5210, "a secondary of 5210 ppm")}, "") == [5210.0, 5210.0]


def test_parse_takes_the_last_valid_dossier():
    text = '{"verdict": "maybe"} then {"verdict": "reject", "reasons": []}'
    assert parse(text)["verdict"] == "reject"
    assert parse("no json here") is None
