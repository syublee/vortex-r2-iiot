import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
AGENT_MANAGER_PATH = REPO_ROOT / "ai_scientist" / "treesearch" / "agent_manager.py"
BACKUP_DIR = Path(__file__).parent / "stage_goals_backup"
PROMPTS_DIR = Path(__file__).parent / "prompts"


def collect_results(experiment_dir: str) -> dict:
    """Parse logs/0-run/stage_*/journal.json -> per-stage metrics."""
    results = {}
    logs_dir = Path(experiment_dir) / "logs" / "0-run"
    if not logs_dir.exists():
        return results

    for stage_dir in sorted(p for p in logs_dir.iterdir() if p.is_dir()):
        journal_path = stage_dir / "journal.json"
        if not journal_path.exists():
            results[stage_dir.name] = {"skipped": True}
            continue

        journal = json.loads(journal_path.read_text())
        nodes = journal.get("nodes", [])
        if not nodes:
            results[stage_dir.name] = {"skipped": True}
            continue

        buggy_count = sum(1 for n in nodes if n.get("is_buggy", True))
        good_nodes = [n for n in nodes if not n.get("is_buggy", True)]
        buggy_rate = buggy_count / len(nodes)

        best_metric: Optional[float] = None
        for node in good_nodes:
            val = _extract_metric_value(node)
            if val is not None:
                best_metric = val if best_metric is None else max(best_metric, val)

        results[stage_dir.name] = {
            "best_metric": best_metric,
            "buggy_rate": buggy_rate,
            "good_nodes": len(good_nodes),
            "skipped": False,
        }
    return results


def _extract_metric_value(node: dict) -> Optional[float]:
    """Extract a single float from the node's metric.value field."""
    try:
        data = node["metric"]["value"]["metric_names"][0]["data"]
        return float(data[0]["final_value"])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def gate(patches: list, current_goals: str) -> list:
    """Keep only patches that are safe to apply."""
    valid = []
    for patch in patches:
        op = patch.get("op")
        old = patch.get("old", "")
        new = patch.get("new", "")

        if op in ("replace", "delete") and old not in current_goals:
            continue

        if op == "replace":
            simulated = current_goals.replace(old, new, 1)
        elif op == "delete":
            simulated = current_goals.replace(old, "", 1)
        elif op == "add":
            simulated = current_goals + "\n" + new
        else:
            continue

        ratio = len(simulated) / max(len(current_goals), 1)
        if not (0.70 <= ratio <= 1.30):
            continue

        valid.append(patch)
    return valid


def apply(patches: list, agent_manager_path: str, backup_dir: str,
          dry_run: bool = False) -> None:
    """Backup original -> apply patches -> write -> print unified diff."""
    import difflib

    path = Path(agent_manager_path)
    content = path.read_text()

    if not dry_run:
        bdir = Path(backup_dir)
        bdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        (bdir / f"{ts}_stage_goals.md").write_text(content)

    patched = content
    for patch in patches:
        op = patch["op"]
        old = patch.get("old", "")
        new = patch.get("new", "")
        if op == "replace":
            patched = patched.replace(old, new, 1)
        elif op == "delete":
            patched = patched.replace(old, "", 1)
        elif op == "add":
            patched = patched + "\n" + new

    diff = list(difflib.unified_diff(
        content.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile="agent_manager.py (before)",
        tofile="agent_manager.py (after)",
    ))
    print("".join(diff) if diff else "(no diff)")

    if not dry_run:
        path.write_text(patched)


def reflect(current_goals: str, results: dict, model: str) -> str:
    """Call Claude to identify which Stage Goals caused failures."""
    from ai_scientist.treesearch.backend import query

    template = (PROMPTS_DIR / "reflect.md").read_text()
    user_message = template.format(
        current_goals=current_goals,
        results=json.dumps(results, indent=2),
    )
    return query(system_message=None, user_message=user_message, model=model)


def propose_patches(reflection: str, current_goals: str, model: str) -> list:
    """Call Claude to propose structured JSON patches."""
    from ai_scientist.treesearch.backend import query, FunctionSpec

    template = (PROMPTS_DIR / "patch.md").read_text()
    user_message = template.format(
        reflection=reflection,
        current_goals=current_goals,
    )

    func_spec = FunctionSpec(
        name="propose_patches",
        description="Return a list of text patches for Stage Goals",
        json_schema={
            "type": "object",
            "properties": {
                "patches": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "properties": {
                            "stage":  {"type": "string"},
                            "op":     {"type": "string", "enum": ["replace", "add", "delete"]},
                            "old":    {"type": "string"},
                            "new":    {"type": "string"},
                        },
                        "required": ["stage", "op", "old"],
                    },
                }
            },
            "required": ["patches"],
        },
    )

    result = query(
        system_message=None,
        user_message=user_message,
        model=model,
        func_spec=func_spec,
    )
    return result.get("patches", [])
