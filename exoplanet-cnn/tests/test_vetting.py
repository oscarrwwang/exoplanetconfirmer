import numpy as np
import pandas as pd
import pytest

from src.inference import vetting as vt

CLEAN = {
    "bls_status": "ok", "bls_snr": 40.0, "bls_odd_even_sigma": 0.5, "bls_secondary_snr": 0.2,
    "bls_depth_ppm": 800.0, "bls_harmonic_dll": -500.0, "bls_n_transits": 12,
    "bls_max_transit_ll_frac": 0.1, "local_v_shape": 1.1, "global_min_oot": -0.1,
    "local_asymmetry": 0.02, "local_oot_std": 0.05,
}


def codes(features):
    return [f.code for f in vt.vet(features)]


def test_clean_planet_has_no_flags():
    assert codes(CLEAN) == []


@pytest.mark.parametrize("key,value,code", [
    ("bls_odd_even_sigma", 6.0, "ODD_EVEN_DEPTH"),
    ("bls_secondary_snr", 9.0, "SECONDARY_ECLIPSE"),
    ("bls_depth_ppm", 80_000.0, "TOO_DEEP"),
    ("bls_harmonic_dll", 12.0, "SINUSOIDAL"),
    ("local_v_shape", 1.45, "V_SHAPED"),
    ("bls_snr", 4.0, "LOW_SNR"),
    ("bls_n_transits", 2, "FEW_TRANSITS"),
    ("bls_max_transit_ll_frac", 0.8, "SINGLE_EVENT"),
    ("global_min_oot", -0.7, "OUT_OF_TRANSIT_DIPS"),
    ("local_asymmetry", -0.4, "ASYMMETRIC"),
    ("local_oot_std", 0.5, "NOISY_FOLD"),
    ("local_coverage", 0.2, "LOW_COVERAGE"),
    ("transit_evidence_share", 0.1, "MODEL_OFF_TRANSIT"),
])
def test_each_check_fires_alone(key, value, code):
    assert codes({**CLEAN, key: value}) == [code]
    flag = vt.vet({**CLEAN, key: value})[0]
    assert flag.severity == vt.MESSAGES[code][0] and flag.message


def test_missing_values_skip_checks_and_failed_bls_is_reported():
    assert codes({**CLEAN, "bls_odd_even_sigma": np.nan, "local_v_shape": None}) == []
    failed = {**CLEAN, "bls_status": "failed", "bls_odd_even_sigma": 50.0}
    assert codes(failed) == ["NO_BLS_FIT"]
    assert codes({**CLEAN, "bls_status": np.nan}) == ["NO_BLS_FIT"]


def test_confidence_score_is_symmetric_around_threshold():
    assert vt.confidence_score(0.3, 0.3) == 0.0
    assert vt.confidence_score(1.0, 0.3) == pytest.approx(1.0)
    assert vt.confidence_score(0.0, 0.3) == pytest.approx(1.0)
    assert vt.confidence_score(0.65, 0.3) == pytest.approx(0.5)
    assert vt.confidence_level(0.7) == "high" and vt.confidence_level(0.1) == "low"


def test_decide_dispositions():
    fp_flag = vt.vet({**CLEAN, "bls_secondary_snr": 10.0})
    assert vt.decide(0.95, 0.5, [])["disposition"] == "PLANET CANDIDATE"
    review = vt.decide(0.95, 0.5, fp_flag)
    assert review["disposition"] == "NEEDS REVIEW"
    assert review["reason_codes"] == ["SECONDARY_ECLIPSE"]
    assert vt.decide(0.05, 0.5, [])["disposition"] == "FALSE POSITIVE"
    assert vt.decide(0.05, 0.5, fp_flag)["disposition"] == "FALSE POSITIVE"
    low = vt.decide(0.52, 0.5, [])
    assert low["disposition"] == "NEEDS REVIEW" and "MODEL_LOW_CONFIDENCE" in low["reason_codes"]
    # Low confidence below threshold but backed by a false-positive test.
    assert vt.decide(0.48, 0.5, fp_flag)["disposition"] == "FALSE POSITIVE"


def test_flag_summary_counts_per_class():
    feats = pd.DataFrame([
        {"kepoi_name": "a", **CLEAN},
        {"kepoi_name": "b", **CLEAN, "bls_secondary_snr": 10.0},
        {"kepoi_name": "c", **CLEAN, "bls_secondary_snr": 10.0, "local_v_shape": 1.5},
    ])
    table = vt.vet_table(feats, np.array([0.9, 0.1, 0.2]), 0.5)
    assert table.disposition.tolist() == ["PLANET CANDIDATE", "FALSE POSITIVE", "FALSE POSITIVE"]
    summary = vt.flag_summary(table, np.array([1, 0, 0])).set_index("code")
    assert summary.loc["SECONDARY_ECLIPSE", "flagged_false_positives"] == 2
    assert summary.loc["SECONDARY_ECLIPSE", "flagged_planets"] == 0
    assert summary.loc["V_SHAPED", "fp_rate_flagged"] == pytest.approx(0.5)
