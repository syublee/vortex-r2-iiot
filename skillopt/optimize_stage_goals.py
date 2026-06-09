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


def main(
    experiment_dir: str,
    model: str = "claude-code/opus",
    dry_run: bool = False,
    agent_manager_path: str = str(AGENT_MANAGER_PATH),
    backup_dir: str = str(BACKUP_DIR),
) -> None:
    results = collect_results(experiment_dir)
    if not results:
        print("SkillOpt: no stage results found — nothing to optimize.")
        return

    non_skipped = [r for r in results.values() if not r.get("skipped")]
    if non_skipped and all(r.get("buggy_rate", 1.0) == 0.0 for r in non_skipped):
        print("SkillOpt: all stages have buggy_rate=0 — already optimal, no changes.")
        return

    current_goals = Path(agent_manager_path).read_text()

    try:
        print("SkillOpt: running reflect step...")
        reflection = reflect(current_goals, results, model)
        print("SkillOpt: running patch proposal step...")
        patches = propose_patches(reflection, current_goals, model)
    except Exception as exc:
        print(f"SkillOpt: Claude call failed ({exc}) — agent_manager.py unchanged.",
              file=sys.stderr)
        return

    valid = gate(patches, current_goals)
    if not valid:
        print("SkillOpt: no valid improvements identified — agent_manager.py unchanged.")
        return

    apply(valid, agent_manager_path, backup_dir, dry_run=dry_run)
    if dry_run:
        print("SkillOpt: dry-run complete — agent_manager.py unchanged.")
    else:
        print(f"SkillOpt: {len(valid)} patch(es) applied to agent_manager.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SkillOpt: improve Stage Goals from BFTS experiment results"
    )
    parser.add_argument(
        "--experiment-dir", required=True,
        help="Path to experiment directory, e.g. experiments/2026-06-09_..._attempt_0"
    )
    parser.add_argument(
        "--model", default="claude-code/opus",
        help="Model to use for reflect and patch steps (default: claude-code/opus)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the proposed diff without modifying agent_manager.py"
    )
    parser.add_argument(
        "--agent-manager-path", default=str(AGENT_MANAGER_PATH),
        help="Path to agent_manager.py (default: auto-detected from repo root)"
    )
    parser.add_argument(
        "--backup-dir", default=str(BACKUP_DIR),
        help="Directory to store Stage Goals backups"
    )
    args = parser.parse_args()
    main(
        experiment_dir=args.experiment_dir,
        model=args.model,
        dry_run=args.dry_run,
        agent_manager_path=args.agent_manager_path,
        backup_dir=args.backup_dir,
    )
