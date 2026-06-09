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
