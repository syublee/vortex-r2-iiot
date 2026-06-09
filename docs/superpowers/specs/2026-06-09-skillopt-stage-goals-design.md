# SkillOpt Stage Goals Optimizer — Design Spec

**Date:** 2026-06-09
**Project:** AI Scientist v2 (`/home/syublee/AI-Scientist-v2/`)

---

## Goal

After each AI Scientist experiment completes, automatically analyze BFTS stage results and improve the `agent_manager.py` Stage Goals text so that future experiments perform better.

---

## Architecture

### Core loop (5 steps)

```
실험 완료
    ↓
[1. Collect]  journal.json → per-stage metrics + buggy rates
    ↓
[2. Reflect]  Claude: "What in the Stage Goal was unclear or unrealistic?"
    ↓
[3. Patch]    Claude: structured add/replace/delete patches (JSON schema)
    ↓
[4. Gate]     Validate: old text exists, length within ±30%
    ↓
[5. Apply]    Backup original → update agent_manager.py → print diff
```

### Scope

- **Optimizes:** `main_stage_goals` dict in `ai_scientist/treesearch/agent_manager.py`
- **Signal:** per-stage `best_metric.value` and `buggy_rate` from `journal.json`
- **Trigger:** automatically after `launch_scientist_bfts.py` completes, via `run_scientist.sh`

---

## Files

### New files

| Path | Purpose |
|------|---------|
| `skillopt/optimize_stage_goals.py` | Main optimizer (~200 lines). Entry point: `main(experiment_dir, dry_run)` |
| `skillopt/prompts/reflect.md` | Reflect prompt template — receives stage results, returns natural-language analysis |
| `skillopt/prompts/patch.md` | Patch prompt template — receives reflection + current goals, returns JSON patch list |
| `skillopt/stage_goals_backup/` | Version history of Stage Goals before each patch |

### Modified files

| Path | Change |
|------|--------|
| `run_scientist.sh` | Add 3-line block after `launch_scientist_bfts.py` to call optimizer |

---

## Component Detail

### `optimize_stage_goals.py`

Five pure functions, called in sequence:

```python
def collect_results(experiment_dir: str) -> dict:
    """
    Parse logs/0-run/stage_*/journal.json.
    Returns: {stage_name: {best_metric, buggy_rate, good_nodes, skipped}}
    Missing journal.json → stage marked skipped, not an error.
    """

def reflect(current_goals: str, results: dict, model: str) -> str:
    """
    Call Claude with reflect.md template + results JSON.
    Returns: natural-language analysis of which Goal text caused failures.
    Raises on Claude error (caller catches, aborts without patching).
    """

def propose_patches(reflection: str, current_goals: str, model: str) -> list[dict]:
    """
    Call Claude with patch.md template.
    Forces JSON schema: [{stage, op, old, new}]
    op in {"replace", "add", "delete"}
    Returns parsed list; empty list is valid (no changes needed).
    """

def gate(patches: list[dict], current_goals: str) -> list[dict]:
    """
    Keep only patches where:
      - op=="replace"/"delete": old text exists verbatim in current_goals
      - Resulting goals length within 70–130% of original
    Silently drop failing patches. Return empty list if none pass.
    """

def apply(patches: list[dict], agent_manager_path: str, backup_dir: str) -> None:
    """
    1. Write backup to backup_dir/{ISO8601}_stage_goals.md
    2. Apply patches to main_stage_goals dict (str.replace)
    3. Write updated agent_manager.py
    4. Print unified diff to stdout
    Aborts if backup write fails.
    """
```

### Prompt templates

**`reflect.md`** — instructs Claude to:
- Read the current Stage Goal text
- Read the per-stage results (metric values + buggy rates)
- Identify which stage goals were ambiguous, over-scoped, or caused high bug rates
- Output 3–5 bullet points of specific observations

**`patch.md`** — instructs Claude to:
- Read the reflection output
- Propose minimal text edits as a JSON array
- Each patch: `{stage: "stage3", op: "replace", old: "...", new: "..."}`
- Limit: max 2 patches per stage, max 8 patches total
- Bias toward shorter, simpler goal text

### `run_scientist.sh` addition

After the `python launch_scientist_bfts.py` call:

```bash
if [[ "${SKIP_SKILLOPT:-0}" -ne 1 && -n "$LATEST" ]]; then
  info "SkillOpt: Stage Goals 최적화 실행 → $LATEST"
  python skillopt/optimize_stage_goals.py --experiment-dir "$LATEST" \
      --model "$MODEL" \
    || warn "SkillOpt 실패 — agent_manager.py 미수정"
fi
```

---

## Data Flow

### Input

```
experiments/{timestamp}_attempt_0/
└── logs/0-run/
    ├── stage_initial_implementation/journal.json
    ├── stage_baseline_tuning/journal.json
    ├── stage_creative_research_{n}/journal.json
    └── stage_ablation_studies/journal.json
```

Each `journal.json` provides: node list with `is_buggy`, `metric.value`, `metric.maximize`.

### Intermediate: results dict

```json
{
  "stage_initial_implementation": {
    "best_metric": 0.82,
    "buggy_rate": 0.44,
    "good_nodes": 5,
    "skipped": false
  },
  "stage_creative_research_2": {
    "best_metric": null,
    "buggy_rate": 1.0,
    "good_nodes": 0,
    "skipped": false
  }
}
```

### Intermediate: patch list

```json
[
  {
    "stage": "stage_creative_research",
    "op": "replace",
    "old": "Implement methodological improvements across 3+ datasets",
    "new": "Start with 1 working dataset. Expand to a second only after the first runs error-free."
  }
]
```

### Output

- `agent_manager.py` `main_stage_goals` dict updated in place
- `skillopt/stage_goals_backup/2026-06-09T14:32:00_stage_goals.md` saved
- Unified diff printed to stdout

---

## Error Handling

| Situation | Behavior |
|-----------|----------|
| `journal.json` missing for a stage | Stage marked `skipped=true`, processing continues |
| All stages `buggy_rate == 0` | Print "already optimal", exit without patching |
| Claude call fails (any step) | Print warning, exit without modifying `agent_manager.py` |
| Gate passes 0 patches | Print "no improvements identified", exit cleanly |
| Backup directory write fails | Abort before patching (safety first) |

---

## CLI Interface

```bash
# Preview only — no file modifications
python skillopt/optimize_stage_goals.py \
    --experiment-dir experiments/2026-06-09_14-32-00_my_topic_attempt_0 \
    --dry-run

# Normal run (model defaults to claude-code/opus)
python skillopt/optimize_stage_goals.py \
    --experiment-dir experiments/2026-06-09_14-32-00_my_topic_attempt_0

# Skip optimizer in run_scientist.sh
./run_scientist.sh ai_scientist/ideas/my_topic.md --skip-skillopt
```

---

## Success Criteria

- After a Stage3 all-buggy experiment, running the optimizer proposes simpler Stage3 goal text
- `--dry-run` prints diff without touching `agent_manager.py`
- A failed Claude call leaves `agent_manager.py` unchanged
- Backups accumulate in `skillopt/stage_goals_backup/` with ISO8601 timestamps
