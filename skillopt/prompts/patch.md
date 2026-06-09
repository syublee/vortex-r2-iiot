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
