import json
import os
import pytest
from pathlib import Path


def _make_experiment(tmp_path, stages: dict) -> str:
    """Helper: build a fake experiment dir with journal.json files."""
    for stage_name, nodes_spec in stages.items():
        stage_dir = tmp_path / "logs" / "0-run" / stage_name
        stage_dir.mkdir(parents=True)
        nodes = []
        for spec in nodes_spec:
            metric_value = None
            if spec.get("metric_names_data"):
                metric_value = {
                    "metric_names": [
                        {"data": [{"final_value": v} for v in spec["metric_names_data"]]}
                    ]
                }
            nodes.append({
                "is_buggy": spec["is_buggy"],
                "metric": {"value": metric_value, "maximize": True},
            })
        (stage_dir / "journal.json").write_text(json.dumps({"nodes": nodes}))
    return str(tmp_path)


def test_collect_results_buggy_rate(tmp_path):
    from skillopt.optimize_stage_goals import collect_results

    exp_dir = _make_experiment(tmp_path, {
        "stage_1_initial_implementation": [
            {"is_buggy": False, "metric_names_data": [0.82]},
            {"is_buggy": True,  "metric_names_data": []},
            {"is_buggy": True,  "metric_names_data": []},
        ]
    })
    results = collect_results(exp_dir)
    key = "stage_1_initial_implementation"
    assert key in results
    assert results[key]["buggy_rate"] == pytest.approx(2 / 3)
    assert results[key]["good_nodes"] == 1
    assert results[key]["skipped"] is False


def test_collect_results_best_metric(tmp_path):
    from skillopt.optimize_stage_goals import collect_results

    exp_dir = _make_experiment(tmp_path, {
        "stage_1_initial_implementation": [
            {"is_buggy": False, "metric_names_data": [0.70]},
            {"is_buggy": False, "metric_names_data": [0.90]},
        ]
    })
    results = collect_results(exp_dir)
    key = "stage_1_initial_implementation"
    assert results[key]["best_metric"] == pytest.approx(0.90)


def test_collect_results_missing_journal_marked_skipped(tmp_path):
    from skillopt.optimize_stage_goals import collect_results

    stage_dir = tmp_path / "logs" / "0-run" / "stage_1_initial_implementation"
    stage_dir.mkdir(parents=True)
    # No journal.json written
    results = collect_results(str(tmp_path))
    assert results["stage_1_initial_implementation"]["skipped"] is True


def test_collect_results_all_buggy_best_metric_is_none(tmp_path):
    from skillopt.optimize_stage_goals import collect_results

    exp_dir = _make_experiment(tmp_path, {
        "stage_3_creative_research": [
            {"is_buggy": True, "metric_names_data": []},
            {"is_buggy": True, "metric_names_data": []},
        ]
    })
    results = collect_results(str(tmp_path))
    assert results["stage_3_creative_research"]["best_metric"] is None
    assert results["stage_3_creative_research"]["buggy_rate"] == pytest.approx(1.0)
