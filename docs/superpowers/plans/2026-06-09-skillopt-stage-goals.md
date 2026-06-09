# SkillOpt Stage Goals Optimizer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After each AI Scientist experiment completes, analyze BFTS stage results with Claude and automatically patch the `agent_manager.py` Stage Goals text to improve future runs.

**Architecture:** A standalone `skillopt/optimize_stage_goals.py` module with five pure functions (collect → reflect → propose_patches → gate → apply) called in sequence. Claude is called twice (reflect + patch) via the existing `ai_scientist.treesearch.backend.query` interface. The result is a unified diff printed to stdout and — unless `--dry-run` — the Stage Goals text updated in place with a timestamped backup.

**Tech Stack:** Python 3.10+, existing `ai_scientist.treesearch.backend.query` + `FunctionSpec`, `pathlib`, `difflib`, `argparse`. No new dependencies.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `skillopt/__init__.py` | Package marker (empty) |
| Create | `skillopt/optimize_stage_goals.py` | Five pure functions + CLI entry point |
| Create | `skillopt/prompts/reflect.md` | Reflect prompt template |
| Create | `skillopt/prompts/patch.md` | Patch proposal prompt template |
| Create | `skillopt/stage_goals_backup/.gitkeep` | Version-controlled placeholder for backup dir |
| Create | `tests/test_optimize_stage_goals.py` | Unit tests (TDD) |
| Modify | `run_scientist.sh` | Add `--skip-skillopt` flag + post-run optimizer call |

---

## Task 1: Project Scaffold

**Files:**
- Create: `skillopt/__init__.py`
- Create: `skillopt/stage_goals_backup/.gitkeep`
- Create: `tests/` (if not already present)

- [ ] **Step 1: Create the skillopt package directory**

```bash
cd /home/syublee/AI-Scientist-v2
mkdir -p skillopt/prompts skillopt/stage_goals_backup tests
touch skillopt/__init__.py skillopt/stage_goals_backup/.gitkeep
```

- [ ] **Step 2: Verify structure**

```bash
ls skillopt/ tests/
```

Expected output:
```
skillopt/:
__init__.py  prompts/  stage_goals_backup/

tests/:
(empty or existing test files)
```

- [ ] **Step 3: Commit scaffold**

```bash
git add skillopt/ tests/
git commit -m "feat: add skillopt package scaffold"
```

---

## Task 2: `collect_results()` — TDD

**Files:**
- Create: `tests/test_optimize_stage_goals.py`
- Create: `skillopt/optimize_stage_goals.py` (partial — `collect_results` only)

- [ ] **Step 1: Write the failing tests for `collect_results`**

Create `tests/test_optimize_stage_goals.py`:

```python
import json
import os
import pytest
from pathlib import Path


def _make_experiment(tmp_path, stages: dict) -> str:
    """Helper: build a fake experiment dir with journal.json files.

    stages = {
      "stage_1_initial_implementation": [
          {"is_buggy": False, "metric_names_data": [0.82]},
          {"is_buggy": True,  "metric_names_data": []},
      ],
      ...
    }
    """
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
```

- [ ] **Step 2: Run tests — verify they FAIL**

```bash
cd /home/syublee/AI-Scientist-v2
python -m pytest tests/test_optimize_stage_goals.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'skillopt.optimize_stage_goals'`

- [ ] **Step 3: Implement `collect_results` and `_extract_metric_value`**

Create `skillopt/optimize_stage_goals.py`:

```python
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
AGENT_MANAGER_PATH = REPO_ROOT / "ai_scientist" / "treesearch" / "agent_manager.py"
BACKUP_DIR = Path(__file__).parent / "stage_goals_backup"
PROMPTS_DIR = Path(__file__).parent / "prompts"


def collect_results(experiment_dir: str) -> dict:
    """Parse logs/0-run/stage_*/journal.json → per-stage metrics."""
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
```

- [ ] **Step 4: Run tests — verify they PASS**

```bash
python -m pytest tests/test_optimize_stage_goals.py::test_collect_results_buggy_rate \
    tests/test_optimize_stage_goals.py::test_collect_results_best_metric \
    tests/test_optimize_stage_goals.py::test_collect_results_missing_journal_marked_skipped \
    tests/test_optimize_stage_goals.py::test_collect_results_all_buggy_best_metric_is_none \
    -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add skillopt/optimize_stage_goals.py tests/test_optimize_stage_goals.py
git commit -m "feat: implement collect_results with TDD"
```

---

## Task 3: `gate()` — TDD

**Files:**
- Modify: `tests/test_optimize_stage_goals.py` (append tests)
- Modify: `skillopt/optimize_stage_goals.py` (append `gate`)

- [ ] **Step 1: Append gate tests to test file**

Append to `tests/test_optimize_stage_goals.py`:

```python
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
    # add 100 chars → ratio = 107/6 >> 1.30
    patches = [{"stage": "1", "op": "add", "old": "", "new": "x" * 100}]
    assert gate(patches, goals) == []


def test_gate_keeps_delete_op():
    from skillopt.optimize_stage_goals import gate

    goals = "Focus on one dataset only. Do not use three datasets."
    patches = [{"stage": "1", "op": "delete",
                "old": " Do not use three datasets.", "new": ""}]
    result = gate(patches, goals)
    assert len(result) == 1


def test_gate_drops_unknown_op():
    from skillopt.optimize_stage_goals import gate

    goals = "Focus on getting a basic working implementation."
    patches = [{"stage": "1", "op": "mutate", "old": "basic", "new": "simple"}]
    assert gate(patches, goals) == []
```

- [ ] **Step 2: Run new tests — verify they FAIL**

```bash
python -m pytest tests/test_optimize_stage_goals.py -k "gate" -v 2>&1 | head -20
```

Expected: `ImportError` or `AttributeError: module has no attribute 'gate'`

- [ ] **Step 3: Implement `gate` in `optimize_stage_goals.py`**

Append to `skillopt/optimize_stage_goals.py`:

```python
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
```

- [ ] **Step 4: Run all tests — verify PASS**

```bash
python -m pytest tests/test_optimize_stage_goals.py -v
```

Expected: all tests PASSED

- [ ] **Step 5: Commit**

```bash
git add skillopt/optimize_stage_goals.py tests/test_optimize_stage_goals.py
git commit -m "feat: implement gate() with TDD"
```

---

## Task 4: `apply()` — TDD

**Files:**
- Modify: `tests/test_optimize_stage_goals.py` (append tests)
- Modify: `skillopt/optimize_stage_goals.py` (append `apply`)

- [ ] **Step 1: Append apply tests**

Append to `tests/test_optimize_stage_goals.py`:

```python
import difflib


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
    assert "-Focus on basic" in captured.out or "basic" in captured.out
```

- [ ] **Step 2: Run new tests — verify they FAIL**

```bash
python -m pytest tests/test_optimize_stage_goals.py -k "apply" -v 2>&1 | head -20
```

Expected: `AttributeError: module has no attribute 'apply'`

- [ ] **Step 3: Implement `apply` in `optimize_stage_goals.py`**

Append to `skillopt/optimize_stage_goals.py`:

```python
def apply(patches: list, agent_manager_path: str, backup_dir: str,
          dry_run: bool = False) -> None:
    """Backup original → apply patches → write → print unified diff."""
    import difflib

    path = Path(agent_manager_path)
    content = path.read_text()

    if not dry_run:
        bdir = Path(backup_dir)
        bdir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
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
```

- [ ] **Step 4: Run all tests — verify PASS**

```bash
python -m pytest tests/test_optimize_stage_goals.py -v
```

Expected: all tests PASSED

- [ ] **Step 5: Commit**

```bash
git add skillopt/optimize_stage_goals.py tests/test_optimize_stage_goals.py
git commit -m "feat: implement apply() with TDD"
```

---

## Task 5: Prompt Templates

**Files:**
- Create: `skillopt/prompts/reflect.md`
- Create: `skillopt/prompts/patch.md`

- [ ] **Step 1: Create `reflect.md`**

Create `skillopt/prompts/reflect.md`:

```markdown
You are analyzing the results of an automated AI research experiment pipeline (BFTS — Best-First Tree Search) to identify weaknesses in the Stage Goal definitions.

## Current Stage Goals (from agent_manager.py)

{current_goals}

## Experiment Results (JSON — per stage)

```json
{results}
```

## Task

Identify which Stage Goal text contributed to poor experiment performance. Signals of poor performance: high `buggy_rate` (close to 1.0), `best_metric` is null, or stage was skipped.

Output exactly 3–5 bullet points. For each:
- Name the stage (e.g. "stage_3_creative_research")
- Quote the **exact phrase** from the Stage Goal that was ambiguous, over-scoped, or unrealistic given the failure pattern
- One sentence explaining why this phrase likely caused the failure
```

- [ ] **Step 2: Create `patch.md`**

Create `skillopt/prompts/patch.md`:

```markdown
You are proposing minimal text edits to fix Stage Goal definitions in an AI research pipeline.

## Reflection on Current Problems

{reflection}

## Current Stage Goals (full text)

{current_goals}

## Task

Call the `propose_patches` function with a JSON array of patches.

Rules:
- Maximum 2 patches per stage, 8 patches total across all stages
- `op: "replace"` — `old` must be an **exact verbatim substring** of the Current Stage Goals text above (copy-paste exactly, including whitespace)
- `op: "delete"` — same requirement for `old`; the `new` field is ignored
- `op: "add"` — `new` is appended to the end of the current goals text; `old` can be empty
- Prefer shorter and more concrete goal text
- Do NOT rewrite entire stages — make the smallest edit that addresses the identified problem
- If a stage had buggy_rate=1.0, prefer making its goal simpler and more achievable
```

- [ ] **Step 3: Verify templates load correctly**

```bash
python -c "
from pathlib import Path
t = Path('skillopt/prompts/reflect.md').read_text()
print(t.format(current_goals='GOALS', results='{}'))
print('--- OK')
"
```

Expected: prints formatted template ending with `--- OK`

- [ ] **Step 4: Commit**

```bash
git add skillopt/prompts/
git commit -m "feat: add reflect and patch prompt templates"
```

---

## Task 6: `reflect()` and `propose_patches()`

**Files:**
- Modify: `skillopt/optimize_stage_goals.py` (append two functions)

These functions call Claude — no unit tests (would require mocking the LLM). Integration-tested in Task 9.

- [ ] **Step 1: Append `reflect()` to `optimize_stage_goals.py`**

```python
def reflect(current_goals: str, results: dict, model: str) -> str:
    """Call Claude to identify which Stage Goals caused failures."""
    from ai_scientist.treesearch.backend import query

    template = (PROMPTS_DIR / "reflect.md").read_text()
    user_message = template.format(
        current_goals=current_goals,
        results=json.dumps(results, indent=2),
    )
    return query(system_message=None, user_message=user_message, model=model)
```

- [ ] **Step 2: Append `propose_patches()` to `optimize_stage_goals.py`**

```python
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
```

- [ ] **Step 3: Run existing tests to verify nothing broke**

```bash
python -m pytest tests/test_optimize_stage_goals.py -v
```

Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add skillopt/optimize_stage_goals.py
git commit -m "feat: implement reflect() and propose_patches()"
```

---

## Task 7: `main()` + CLI Entry Point

**Files:**
- Modify: `skillopt/optimize_stage_goals.py` (append `main` + `if __name__` block)

- [ ] **Step 1: Append `main()` to `optimize_stage_goals.py`**

```python
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
```

- [ ] **Step 2: Append CLI block to `optimize_stage_goals.py`**

```python
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
```

- [ ] **Step 3: Verify CLI help text works**

```bash
python skillopt/optimize_stage_goals.py --help
```

Expected: prints usage with `--experiment-dir`, `--model`, `--dry-run`, `--agent-manager-path`, `--backup-dir`

- [ ] **Step 4: Run all tests one more time**

```bash
python -m pytest tests/test_optimize_stage_goals.py -v
```

Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add skillopt/optimize_stage_goals.py
git commit -m "feat: add main() entry point and CLI for skillopt optimizer"
```

---

## Task 8: Update `run_scientist.sh`

**Files:**
- Modify: `run_scientist.sh`

- [ ] **Step 1: Add `--skip-skillopt` to the argument parser block**

In `run_scientist.sh`, find the `while [[ $# -gt 0 ]]; do` block and add one line:

```bash
    --skip-skillopt) SKIP_SKILLOPT=1; shift ;;
```

The full block (find at line ~59) should look like:

```bash
SKIP_SKILLOPT=0   # add this near the other defaults (around line 49)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-gen)       MAX_GEN="$2"; shift 2 ;;
    --reflections)   REFL="$2"; shift 2 ;;
    --journal)       JOURNAL="$2"; shift 2 ;;
    --page-limit)    PAGE_LIMIT="$2"; shift 2 ;;
    --num-cite)      NUM_CITE="$2"; shift 2 ;;
    --attempt-id)    ATTEMPT="$2"; shift 2 ;;
    --idea-idx)      IDEA_IDX="$2"; shift 2 ;;
    --skip-ideation) SKIP_IDEATION=1; shift ;;
    --skip-writeup)  SKIP_WRITEUP=1; shift ;;
    --skip-review)   SKIP_REVIEW=1; shift ;;
    --skip-skillopt) SKIP_SKILLOPT=1; shift ;;    # <-- add this line
    --bfts-config)   BFTS_CONFIG="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    -*)              echo "알 수 없는 옵션: $1" >&2; usage; exit 2 ;;
    *)               TOPIC_MD="$1"; shift ;;
  esac
done
```

- [ ] **Step 2: Add the SkillOpt call after the LATEST assignment**

In `run_scientist.sh`, find:

```bash
LATEST="$(ls -dt experiments/*_attempt_"${ATTEMPT}"/ 2>/dev/null | head -1 || true)"
echo "============================================================"
```

Replace with:

```bash
LATEST="$(ls -dt experiments/*_attempt_"${ATTEMPT}"/ 2>/dev/null | head -1 || true)"

if [[ "$SKIP_SKILLOPT" -ne 1 && -n "$LATEST" ]]; then
  info "SkillOpt: Stage Goals 최적화 실행 → $LATEST"
  python skillopt/optimize_stage_goals.py \
    --experiment-dir "$LATEST" \
    --model "$MODEL" \
    || warn "SkillOpt 실패 — agent_manager.py 미수정"
fi

echo "============================================================"
```

- [ ] **Step 3: Verify the script parses correctly**

```bash
bash -n run_scientist.sh && echo "syntax OK"
```

Expected: `syntax OK`

- [ ] **Step 4: Test `--skip-skillopt` flag is accepted**

```bash
./run_scientist.sh --help | grep skip-skillopt
```

Expected: `--skip-skillopt` appears in usage output

- [ ] **Step 5: Commit**

```bash
git add run_scientist.sh
git commit -m "feat: wire skillopt optimizer into run_scientist.sh"
```

---

## Task 9: End-to-End Smoke Test

Verify the optimizer runs correctly against a real (completed) experiment directory.

**Files:** No changes — read-only test using existing experiment data.

- [ ] **Step 1: Find a completed experiment directory**

```bash
ls /home/syublee/AI-Scientist-Energy-v1/experiments/ | tail -5
```

Pick one with multiple stage directories (e.g. `2026-06-08_15-16-22_cross_sector_contingency_gap_attempt_0`).

- [ ] **Step 2: Run optimizer in dry-run mode against the real experiment**

```bash
cd /home/syublee/AI-Scientist-v2
python skillopt/optimize_stage_goals.py \
  --experiment-dir /home/syublee/AI-Scientist-Energy-v1/experiments/2026-06-08_15-16-22_cross_sector_contingency_gap_attempt_0 \
  --dry-run
```

Expected one of:
- Prints a unified diff showing proposed Stage Goal changes
- Prints "SkillOpt: no valid improvements identified"
- Prints "SkillOpt: all stages have buggy_rate=0 — already optimal"

Any of these is a pass — the key check is **no Python exceptions**.

- [ ] **Step 3: Verify `agent_manager.py` was NOT modified (dry-run guard)**

```bash
git diff ai_scientist/treesearch/agent_manager.py
```

Expected: no output (file unchanged)

- [ ] **Step 4: Run optimizer in live mode against the same experiment**

```bash
python skillopt/optimize_stage_goals.py \
  --experiment-dir /home/syublee/AI-Scientist-Energy-v1/experiments/2026-06-08_15-16-22_cross_sector_contingency_gap_attempt_0
```

Expected: either patches applied (diff printed + backup created) or "no improvements identified"

- [ ] **Step 5: If patches were applied — verify backup was created**

```bash
ls skillopt/stage_goals_backup/
```

Expected: one file named `YYYY-MM-DDTHH-MM-SS_stage_goals.md`

- [ ] **Step 6: Run all unit tests one final time**

```bash
python -m pytest tests/test_optimize_stage_goals.py -v
```

Expected: all PASSED

- [ ] **Step 7: Final commit**

```bash
git add skillopt/stage_goals_backup/
git commit -m "test: verify skillopt smoke test passes end-to-end"
```

---

## Self-Review Checklist

**Spec coverage:**
- ✅ `collect_results` → Task 2
- ✅ `reflect` → Task 6
- ✅ `propose_patches` → Task 6
- ✅ `gate` → Task 3
- ✅ `apply` (backup + patch + diff + dry-run) → Task 4
- ✅ Prompt templates → Task 5
- ✅ `main()` + CLI with `--experiment-dir`, `--model`, `--dry-run` → Task 7
- ✅ `run_scientist.sh` integration → Task 8
- ✅ Error handling (Claude failure → unchanged, gate=0 → message, no results → message) → Task 7
- ✅ End-to-end smoke test → Task 9

**Type consistency:**
- `collect_results(experiment_dir: str) -> dict` — used consistently in Tasks 2, 7
- `gate(patches: list, current_goals: str) -> list` — used consistently in Tasks 3, 7
- `apply(patches, agent_manager_path, backup_dir, dry_run=False)` — used consistently in Tasks 4, 7
- `reflect(current_goals, results, model) -> str` — used consistently in Tasks 6, 7
- `propose_patches(reflection, current_goals, model) -> list` — used consistently in Tasks 6, 7
