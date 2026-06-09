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


def test_gate_drops_nonexistent_old_text():
    from skillopt.optimize_stage_goals import gate

    goals = "Focus on getting a basic working implementation."
    patches = [{"stage": "1", "op": "replace", "old": "DOES NOT EXIST", "new": "simple"}]
    assert gate(patches, goals) == []


def test_gate_keeps_valid_replace():
    from skillopt.optimize_stage_goals import gate

    goals = "Focus on getting a basic working implementation."
    patches = [{"stage": "1", "op": "replace",
                "old": "basic working implementation",
                "new": "working implementation"}]
    result = gate(patches, goals)
    assert len(result) == 1
    assert result[0]["new"] == "working implementation"


def test_gate_drops_oversized_patch():
    from skillopt.optimize_stage_goals import gate

    goals = "Short."  # 6 chars
    # add 100 chars -> ratio >> 1.30
    patches = [{"stage": "1", "op": "add", "old": "", "new": "x" * 100}]
    assert gate(patches, goals) == []


def test_gate_keeps_delete_op():
    from skillopt.optimize_stage_goals import gate

    goals = "Focus on one dataset only. Avoid three datasets if possible."
    patches = [{"stage": "1", "op": "delete",
                "old": " if possible", "new": ""}]
    result = gate(patches, goals)
    assert len(result) == 1


def test_gate_drops_unknown_op():
    from skillopt.optimize_stage_goals import gate

    goals = "Focus on getting a basic working implementation."
    patches = [{"stage": "1", "op": "mutate", "old": "basic", "new": "simple"}]
    assert gate(patches, goals) == []


def test_apply_dry_run_does_not_modify_file(tmp_path):
    from skillopt.optimize_stage_goals import apply

    agent_manager = tmp_path / "agent_manager.py"
    agent_manager.write_text("self.main_stage_goals = {1: 'Focus on basic implementation'}")
    backup_dir = tmp_path / "backup"

    patches = [{"op": "replace", "old": "basic implementation", "new": "simple implementation"}]
    apply(patches, str(agent_manager), str(backup_dir), dry_run=True)

    assert "basic implementation" in agent_manager.read_text()
    assert not backup_dir.exists()


def test_apply_creates_backup(tmp_path):
    from skillopt.optimize_stage_goals import apply

    agent_manager = tmp_path / "agent_manager.py"
    agent_manager.write_text("self.main_stage_goals = {1: 'Focus on basic implementation'}")
    backup_dir = tmp_path / "backup"

    patches = [{"op": "replace", "old": "basic implementation", "new": "simple implementation"}]
    apply(patches, str(agent_manager), str(backup_dir), dry_run=False)

    backups = list(backup_dir.iterdir())
    assert len(backups) == 1
    assert backups[0].name.endswith("_stage_goals.md")
    assert "basic implementation" in backups[0].read_text()


def test_apply_modifies_agent_manager(tmp_path):
    from skillopt.optimize_stage_goals import apply

    agent_manager = tmp_path / "agent_manager.py"
    agent_manager.write_text("self.main_stage_goals = {1: 'Focus on basic implementation'}")
    backup_dir = tmp_path / "backup"

    patches = [{"op": "replace", "old": "basic implementation", "new": "simple implementation"}]
    apply(patches, str(agent_manager), str(backup_dir), dry_run=False)

    content = agent_manager.read_text()
    assert "simple implementation" in content
    assert "basic implementation" not in content


def test_apply_delete_op(tmp_path):
    from skillopt.optimize_stage_goals import apply

    agent_manager = tmp_path / "agent_manager.py"
    agent_manager.write_text("Use THREE datasets. Be creative.")
    backup_dir = tmp_path / "backup"

    patches = [{"op": "delete", "old": "THREE datasets. ", "new": ""}]
    apply(patches, str(agent_manager), str(backup_dir), dry_run=False)

    content = agent_manager.read_text()
    assert "THREE" not in content
    assert "Be creative." in content


def test_apply_prints_diff(tmp_path, capsys):
    from skillopt.optimize_stage_goals import apply

    agent_manager = tmp_path / "agent_manager.py"
    agent_manager.write_text("Focus on basic implementation.")
    backup_dir = tmp_path / "backup"

    patches = [{"op": "replace", "old": "basic", "new": "simple"}]
    apply(patches, str(agent_manager), str(backup_dir), dry_run=False)

    captured = capsys.readouterr()
    assert "basic" in captured.out or "simple" in captured.out
