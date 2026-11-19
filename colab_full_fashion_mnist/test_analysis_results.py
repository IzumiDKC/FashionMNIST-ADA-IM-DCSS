import numpy as np
import pytest
import analyze_results as a


def test_scores_invariant_to_cluster_ids():
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])
    assert a.scores_from_predictions(y, pred) == {"ARI": 1., "NMI": 1., "ACC": 1.}
    assert a.scores_from_predictions(np.zeros(5, dtype=int), np.zeros(5, dtype=int))["ARI"] == 1.


def test_exact_sign_flip_constants_and_null():
    positive = a.paired_stats([.01] * 10)
    assert positive["delta_mean_pp"] == pytest.approx(1.)
    assert positive["p_signflip_two_sided"] == 2 / 1024
    assert positive["ci95_low_pp"] == pytest.approx(1.)
    null = a.paired_stats([-.01, .01])
    assert null["delta_mean_pp"] == 0 and null["p_signflip_two_sided"] == 1.
    assert a.paired_stats([.01])["ci95_low_pp"] is None
    assert a.paired_stats([])["delta_mean_pp"] is None


def test_holm_adjust_preserves_original_order():
    assert np.allclose(a.holm_adjust([.01, .04, .03]), [.03, .06, .06])
    assert len(a.holm_adjust([])) == 0


def test_missing_full_trace_not_called_verified_homogeneous():
    env = {"device": "cuda", "gpu": "Tesla T4", "packages": {"torch": "2.14.1"}}
    old = a.environment_info({}, {"environment": env})
    assert not old["strict_single_environment"]
    new = a.environment_info({"execution_trace": [{"environment": env}]}, {})
    assert new["strict_single_environment"]
    mixed = a.environment_info({"execution_trace": [{"environment": env}, {"environment": {**env, "gpu": "A100"}}]}, {})
    assert mixed["mixed_environment"] and not mixed["strict_single_environment"]


def test_duplicate_campaign_needs_explicit_choice(tmp_path):
    for name in ["paper70k_a", "paper70k_b"]:
        path = tmp_path / name
        path.mkdir()
        (path / "manifest.json").write_text('{"config":{"profile":"paper70k"}}')
    with pytest.raises(ValueError, match="SELECTED_RUNS"):
        a.discover_profiles(tmp_path)
    selected = a.discover_profiles(tmp_path, {"paper70k": "paper70k_b"})
    assert selected["paper70k"].name == "paper70k_b"
