# AI-Scientist-Evolve-v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 새 디렉터리 `/home/syublee/AI-Scientist-Evolve-v1/`에 ShinkaEvolve 기반 진화 파이프라인을 구축한다 — 아이디어 진화(Phase 1) → 실험 코드 진화(Phase 2) → 논문 생성(Phase 3).

**Architecture:** Phase 1은 `generate_idea()` 함수를 가진 Python 파일을 ShinkaEvolve로 진화시키고, 별도의 evaluator가 Novelty·Feasibility·Impact를 채점한다. Phase 2는 pandapower 기반 `run_experiment()` runfile을 진화시키고 실행 점수로 선택한다. Phase 3는 AI-Scientist-v2 모듈을 심볼릭 링크로 재활용한다.

**Tech Stack:** shinka-evolve==0.0.7, anthropic SDK, sentence-transformers (임베딩 기반 Novelty), pandapower (Phase 2 시뮬레이션), Python 3.11, conda env `ai_scientist_energy`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create (dir) | `/home/syublee/AI-Scientist-Evolve-v1/` | 프로젝트 루트 |
| Create | `evolve/phase1_idea/seed/initial.py` | 진화 대상 초기 아이디어 프로그램 |
| Create | `evolve/phase1_idea/eval_idea.py` | Phase 1 평가 스크립트 (Novelty+Feasibility+Impact) |
| Create | `evolve/phase1_idea/run_phase1.py` | Phase 1 ShinkaEvolveRunner 실행 진입점 |
| Create | `evolve/phase2_code/scaffold.py` | best idea JSON → 초기 runfile.py 생성 |
| Create | `evolve/phase2_code/eval_code.py` | Phase 2 평가 스크립트 (run_experiment 실행) |
| Create | `evolve/phase2_code/run_phase2.py` | Phase 2 ShinkaEvolveRunner 실행 진입점 |
| Create | `pipeline.py` | Phase 1→2→3 오케스트레이션 |
| Create | `references/README.md` | 참고 자료 업로드 안내 |
| Create | `tools/pandapower.md` | pandapower 툴 소개 |
| Create | `tools/README.md` | 툴 선택 가이드 |
| Create | `tests/test_phase1_evaluator.py` | Phase 1 evaluator 단위 테스트 |
| Create | `tests/test_phase2_evaluator.py` | Phase 2 evaluator 단위 테스트 |
| Create | `tests/test_pipeline.py` | 통합 테스트 (단축 설정) |
| Create | `environment.yml` | conda 환경 스펙 |

> 모든 경로는 프로젝트 루트 `/home/syublee/AI-Scientist-Evolve-v1/` 기준 상대경로.

---

## Task 1: 프로젝트 디렉터리 생성 및 환경 설정

**Files:**
- Create: `/home/syublee/AI-Scientist-Evolve-v1/` (전체 폴더 구조)
- Create: `environment.yml`

- [ ] **Step 1: 디렉터리 구조 생성**

```bash
mkdir -p /home/syublee/AI-Scientist-Evolve-v1
cd /home/syublee/AI-Scientist-Evolve-v1
mkdir -p evolve/phase1_idea/seed
mkdir -p evolve/phase2_code
mkdir -p references/papers references/datasets
mkdir -p tools logs papers workspaces
mkdir -p tests results/phase1 results/phase2
```

- [ ] **Step 2: git 초기화**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git init
echo "results/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo "*.sqlite" >> .gitignore
echo "logs/" >> .gitignore
git add .gitignore
git commit -m "chore: init repo with .gitignore"
```

- [ ] **Step 3: 심볼릭 링크 생성**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
ln -s ../AI-Scientist-v2/ai_scientist ai_scientist
ln -s ../AI-Scientist-v2/data data
```

- [ ] **Step 4: shinka-evolve 설치**

```bash
source /home/syublee/miniconda3/etc/profile.d/conda.sh
conda activate ai_scientist_energy
pip install shinka-evolve==0.0.7
```

Expected: `Successfully installed shinka-evolve-0.0.7`

- [ ] **Step 5: environment.yml 작성**

```yaml
name: ai_scientist_energy
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - shinka-evolve==0.0.7
    - sentence-transformers>=2.7.0
    - anthropic>=0.30.0
    - pandapower>=2.14.0
    - numpy
    - scipy
    - scikit-learn
```

```bash
# environment.yml 위치: /home/syublee/AI-Scientist-Evolve-v1/environment.yml
```

- [ ] **Step 6: sentence-transformers 설치 확인**

```bash
python3 -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('all-MiniLM-L6-v2'); print('OK', m.encode(['test']).shape)"
```

Expected: `OK (1, 384)`

- [ ] **Step 7: 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add environment.yml
git commit -m "chore: add environment.yml with shinka-evolve dependency"
```

---

## Task 2: Phase 1 — 초기 아이디어 시드 프로그램

**Files:**
- Create: `evolve/phase1_idea/seed/initial.py`

ShinkaEvolve가 진화시키는 대상 파일. `generate_idea()` 함수 안의 `# === EVOLVE-BLOCK ===` 구간이 변이된다.

- [ ] **Step 1: initial.py 작성**

```python
# evolve/phase1_idea/seed/initial.py
"""초기 아이디어 시드 — ShinkaEvolve가 EVOLVE-BLOCK 구간을 변이시킨다."""


def generate_idea() -> dict:
    """에너지 도메인 연구 아이디어를 반환한다."""

    # === EVOLVE-BLOCK START: problem ===
    problem = (
        "IIoT 산업 환경에서 고장 진단은 라벨 희소성 문제에 직면한다. "
        "수동 라벨링 비용이 높아 전체 데이터의 1-5%만 라벨이 부여되며, "
        "기존 CNN은 저라벨 환경에서 과적합되기 쉽다."
    )
    # === EVOLVE-BLOCK END ===

    # === EVOLVE-BLOCK START: hypothesis ===
    hypothesis = (
        "표 형식 센서 데이터를 이미지로 변환할 때 반상관 센서를 이미지 중심에 "
        "배치하면 Hessian 블록이 잘 조건화되어 평탄 수렴을 유도하고, "
        "저라벨 환경에서 CNN 일반화가 향상된다."
    )
    # === EVOLVE-BLOCK END ===

    # === EVOLVE-BLOCK START: method ===
    method = (
        "반상관 그래프(w_ij = 1 - |r_ij|)를 구축하고 force-directed 레이아웃으로 "
        "독립 센서를 이미지 중심에 배치한다. Archimedean 나선형 재정렬을 적용하고 "
        "ResNet-18로 분류한다."
    )
    # === EVOLVE-BLOCK END ===

    # === EVOLVE-BLOCK START: metrics ===
    metrics = [
        "1% 라벨 정확도",
        "Macro-F1",
        "Wilcoxon p-value vs. 기존 baselines",
    ]
    # === EVOLVE-BLOCK END ===

    return {
        "Title": "Vortex-R2: 관계 정규화 표-이미지 변환을 통한 IIoT 고장 진단",
        "Keywords": ["IIoT", "fault_diagnosis", "semi-supervised", "tabular-to-image"],
        "TL;DR": "반상관 센서 배치로 저라벨 IIoT CNN 성능 향상",
        "Problem": problem,
        "Hypothesis": hypothesis,
        "Method": method,
        "Metrics": metrics,
        "combined_score": 0.0,
    }
```

- [ ] **Step 2: 실행 확인**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -c "
from evolve.phase1_idea.seed.initial import generate_idea
idea = generate_idea()
print('Keys:', list(idea.keys()))
print('Problem[:50]:', idea['Problem'][:50])
"
```

Expected: `Keys: ['Title', 'Keywords', 'TL;DR', 'Problem', 'Hypothesis', 'Method', 'Metrics', 'combined_score']`

- [ ] **Step 3: 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add evolve/phase1_idea/seed/initial.py
git commit -m "feat(phase1): add initial idea seed program with EVOLVE-BLOCKs"
```

---

## Task 3: Phase 1 — 아이디어 평가 스크립트

**Files:**
- Create: `evolve/phase1_idea/eval_idea.py`
- Test: `tests/test_phase1_evaluator.py`

ShinkaEvolve가 subprocess로 호출하는 평가 스크립트. `--program_path`의 파일을 로드해 `generate_idea()`를 호출하고, Novelty·Feasibility·Impact를 채점한 뒤 `results_dir/metrics.json`에 기록한다.

- [ ] **Step 1: 테스트 먼저 작성**

```python
# tests/test_phase1_evaluator.py
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_dummy_program(tmp_dir: str, problem: str = "test problem") -> str:
    """테스트용 아이디어 프로그램 파일을 생성하고 경로를 반환."""
    path = os.path.join(tmp_dir, "dummy.py")
    code = f'''
def generate_idea():
    return {{
        "Title": "Test Idea",
        "Keywords": ["energy"],
        "TL;DR": "short",
        "Problem": "{problem}",
        "Hypothesis": "if X then Y",
        "Method": "use Z algorithm",
        "Metrics": ["accuracy"],
        "combined_score": 0.0,
    }}
'''
    Path(path).write_text(code)
    return path


def test_score_novelty_returns_float():
    from evolve.phase1_idea.eval_idea import score_novelty

    with tempfile.TemporaryDirectory() as tmp:
        prog = _make_dummy_program(tmp)
        import importlib.util
        spec = importlib.util.spec_from_file_location("p", prog)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        idea = m.generate_idea()

        score = score_novelty(idea, seed_dir=str(Path(__file__).parent.parent / "evolve/phase1_idea/seed"))
        assert isinstance(score, float), f"Expected float, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Out of range: {score}"


def test_score_feasibility_returns_float(monkeypatch):
    """LLM 호출을 stub하여 비용 없이 테스트."""
    from evolve.phase1_idea import eval_idea

    monkeypatch.setattr(eval_idea, "_call_llm_score", lambda prompt: 0.7)

    with tempfile.TemporaryDirectory() as tmp:
        prog = _make_dummy_program(tmp)
        import importlib.util
        spec = importlib.util.spec_from_file_location("p", prog)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        idea = m.generate_idea()

        score = eval_idea.score_feasibility(idea, references_dir="references/papers")
        assert 0.0 <= score <= 1.0, f"Out of range: {score}"


def test_eval_writes_metrics_json():
    """eval_idea 스크립트가 metrics.json을 올바르게 작성하는지 확인."""
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        prog = _make_dummy_program(tmp, problem="energy optimization challenge")
        results_dir = os.path.join(tmp, "results")
        result = subprocess.run(
            [
                "python3",
                "evolve/phase1_idea/eval_idea.py",
                "--program_path", prog,
                "--results_dir", results_dir,
                "--skip_llm",  # LLM 호출 건너뜀 (테스트 전용 플래그)
            ],
            capture_output=True, text=True,
            cwd="/home/syublee/AI-Scientist-Evolve-v1",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        metrics_path = os.path.join(results_dir, "metrics.json")
        assert os.path.exists(metrics_path), "metrics.json not created"
        metrics = json.loads(Path(metrics_path).read_text())
        assert "combined_score" in metrics
        assert isinstance(metrics["combined_score"], float)
        assert 0.0 <= metrics["combined_score"] <= 1.0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -m pytest tests/test_phase1_evaluator.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'evolve.phase1_idea.eval_idea'`

- [ ] **Step 3: eval_idea.py 구현**

```python
# evolve/phase1_idea/eval_idea.py
"""Phase 1 아이디어 평가 스크립트.

ShinkaEvolve LocalJobConfig 가 subprocess로 호출:
  python eval_idea.py --program_path <path> --results_dir <dir> [--skip_llm]

출력: results_dir/metrics.json  {"combined_score": float, "novelty": float, ...}
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


# ─── 임베딩 기반 Novelty ────────────────────────────────────────────────────

def _load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def _idea_to_text(idea: dict) -> str:
    return " ".join([
        idea.get("Problem", ""),
        idea.get("Hypothesis", ""),
        idea.get("Method", ""),
    ])


def score_novelty(idea: dict, seed_dir: str) -> float:
    """기존 시드 아이디어 대비 코사인 비유사도 (1 = 완전히 새로움)."""
    model = _load_model()
    idea_text = _idea_to_text(idea)
    idea_emb = model.encode([idea_text])

    seed_texts = []
    for f in Path(seed_dir).glob("*.py"):
        try:
            spec = importlib.util.spec_from_file_location("seed", f)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            if hasattr(m, "generate_idea"):
                seed_idea = m.generate_idea()
                seed_texts.append(_idea_to_text(seed_idea))
        except Exception:
            pass
    # JSON 시드도 지원
    for f in Path(seed_dir).glob("*.json"):
        try:
            seed_idea = json.loads(f.read_text())
            seed_texts.append(_idea_to_text(seed_idea))
        except Exception:
            pass

    if not seed_texts:
        return 1.0

    from sklearn.metrics.pairwise import cosine_similarity
    seed_embs = model.encode(seed_texts)
    sims = cosine_similarity(idea_emb, seed_embs)[0]
    return float(1.0 - float(sims.max()))


# ─── LLM 채점 ────────────────────────────────────────────────────────────────

def _call_llm_score(prompt: str) -> float:
    """Anthropic API 호출 → 0.0~1.0 점수 반환."""
    import anthropic
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip().split()[0]
    score = float(raw)
    return max(0.0, min(1.0, score))


def _load_references(references_dir: str) -> str:
    """references/papers/ 내 .md 파일을 합쳐서 반환 (최대 4000자)."""
    texts = []
    for f in sorted(Path(references_dir).glob("*.md")):
        texts.append(f.read_text()[:1000])
    return "\n\n".join(texts)[:4000] if texts else "(참고 자료 없음)"


def score_feasibility(idea: dict, references_dir: str) -> float:
    """에너지 도메인 실현 가능성 LLM 채점 (0~1)."""
    refs = _load_references(references_dir)
    prompt = (
        "당신은 에너지 시스템 연구 전문가입니다. "
        "아래 연구 아이디어의 에너지 도메인 실현 가능성을 0.0~1.0으로 채점하세요. "
        "숫자만 출력하세요.\n\n"
        f"참고 자료:\n{refs}\n\n"
        f"문제: {idea.get('Problem', '')}\n"
        f"가설: {idea.get('Hypothesis', '')}\n"
        f"방법: {idea.get('Method', '')}\n"
    )
    return _call_llm_score(prompt)


def score_impact(idea: dict) -> float:
    """연구 기여도 LLM 채점 (0~1)."""
    prompt = (
        "당신은 에너지 시스템 연구 전문가입니다. "
        "아래 연구 아이디어의 학술적 기여도를 0.0~1.0으로 채점하세요. "
        "숫자만 출력하세요.\n\n"
        f"문제: {idea.get('Problem', '')}\n"
        f"가설: {idea.get('Hypothesis', '')}\n"
        f"방법: {idea.get('Method', '')}\n"
        f"지표: {idea.get('Metrics', '')}\n"
    )
    return _call_llm_score(prompt)


# ─── 프로그램 로더 ─────────────────────────────────────────────────────────────

def _load_program(program_path: str) -> Any:
    spec = importlib.util.spec_from_file_location("program", program_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument(
        "--skip_llm", action="store_true",
        help="LLM 호출 생략 (테스트용). novelty 0.5, feasibility 0.5, impact 0.5 고정."
    )
    args = parser.parse_args()

    # 아이디어 생성
    module = _load_program(args.program_path)
    idea = module.generate_idea()

    # 채점
    seed_dir = str(Path(__file__).parent / "seed")
    refs_dir = str(Path(__file__).parent.parent.parent / "references/papers")

    novelty = score_novelty(idea, seed_dir=seed_dir)

    if args.skip_llm:
        feasibility = 0.5
        impact = 0.5
    else:
        feasibility = score_feasibility(idea, references_dir=refs_dir)
        impact = score_impact(idea)

    combined = 0.35 * novelty + 0.35 * feasibility + 0.30 * impact

    metrics = {
        "combined_score": round(combined, 4),
        "novelty": round(novelty, 4),
        "feasibility": round(feasibility, 4),
        "impact": round(impact, 4),
    }

    # 결과 저장
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[eval_idea] combined_score={combined:.4f} | {metrics}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: `evolve/__init__.py` 및 `evolve/phase1_idea/__init__.py` 생성**

```bash
touch /home/syublee/AI-Scientist-Evolve-v1/evolve/__init__.py
touch /home/syublee/AI-Scientist-Evolve-v1/evolve/phase1_idea/__init__.py
touch /home/syublee/AI-Scientist-Evolve-v1/evolve/phase1_idea/seed/__init__.py
touch /home/syublee/AI-Scientist-Evolve-v1/evolve/phase2_code/__init__.py
touch /home/syublee/AI-Scientist-Evolve-v1/tests/__init__.py
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -m pytest tests/test_phase1_evaluator.py -v
```

Expected: 3 tests pass (LLM 테스트는 stub으로 대체)

- [ ] **Step 6: 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add evolve/phase1_idea/eval_idea.py evolve/__init__.py evolve/phase1_idea/__init__.py evolve/phase1_idea/seed/__init__.py evolve/phase2_code/__init__.py tests/test_phase1_evaluator.py tests/__init__.py
git commit -m "feat(phase1): add idea evaluator with novelty+feasibility+impact scoring"
```

---

## Task 4: Phase 1 — ShinkaEvolveRunner 실행 스크립트

**Files:**
- Create: `evolve/phase1_idea/run_phase1.py`

- [ ] **Step 1: run_phase1.py 작성**

```python
# evolve/phase1_idea/run_phase1.py
"""Phase 1: 아이디어 진화 실행.

사용법:
  python evolve/phase1_idea/run_phase1.py \
      --generations 20 --islands 3 --max_eval_jobs 2

결과: results/phase1/ 디렉터리에 SQLite DB + 최고 아이디어 파일 저장
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import shinka
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch.scheduler import LocalJobConfig

ROOT = Path(__file__).parent.parent.parent  # /home/syublee/AI-Scientist-Evolve-v1
EVAL_SCRIPT = str(Path(__file__).parent / "eval_idea.py")
SEED_PROGRAM = str(Path(__file__).parent / "seed/initial.py")

TASK_SYS_MSG = """당신은 에너지 도메인 연구 아이디어 전문가입니다.
주어진 Python 함수의 EVOLVE-BLOCK 구간을 변이시켜 더 참신하고 실현 가능한
에너지 연구 아이디어를 생성하세요.

규칙:
1. EVOLVE-BLOCK 마커 외부는 절대 수정하지 마세요.
2. generate_idea() 함수 시그니처를 유지하세요.
3. 반환 dict의 키(Title, Keywords, TL;DR, Problem, Hypothesis, Method, Metrics, combined_score)를 유지하세요.
4. 에너지 시스템(전력망, 열, 가스, 재생에너지) 또는 IIoT 도메인에 집중하세요.
5. 구체적이고 측정 가능한 가설을 제시하세요.
"""


def run(generations: int, islands: int, max_eval_jobs: int, results_dir: str) -> None:
    evo_config = EvolutionConfig(
        task_sys_msg=TASK_SYS_MSG,
        language="python",
        num_generations=generations,
        patch_types=["diff", "full"],
        patch_type_probs=[0.7, 0.3],
        llm_models=["claude-sonnet-4-6"],
        llm_kwargs={"temperatures": [0.7, 1.0], "max_tokens": 4096},
        embedding_model=None,   # novelty는 evaluator 내부에서 sentence-transformers로 처리
        init_program_path=SEED_PROGRAM,
        results_dir=results_dir,
        meta_rec_interval=5,
    )

    db_config = DatabaseConfig(
        db_path=os.path.join(results_dir, "evolution_db.sqlite"),
        num_islands=islands,
        archive_size=30,
        exploitation_ratio=0.3,
        elite_selection_ratio=0.3,
    )

    job_config = LocalJobConfig(
        eval_program_path=EVAL_SCRIPT,
        conda_env="ai_scientist_energy",
    )

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=max_eval_jobs,
        max_proposal_jobs=max_eval_jobs + 1,
        max_db_workers=2,
        verbose=True,
    )
    runner.run()

    print(f"\n[Phase 1 완료] 결과: {results_dir}")
    _export_top_k(results_dir, k=5)


def _export_top_k(results_dir: str, k: int) -> None:
    """DB에서 상위 k개 아이디어를 evolved_ideas/ 에 JSON으로 저장."""
    import sqlite3
    db_path = os.path.join(results_dir, "evolution_db.sqlite")
    if not os.path.exists(db_path):
        print("[warn] DB 없음, export 건너뜀")
        return

    out_dir = ROOT / "evolved_ideas"
    out_dir.mkdir(exist_ok=True)

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT program_str, combined_score FROM programs ORDER BY combined_score DESC LIMIT ?", (k,)
    ).fetchall()
    conn.close()

    for i, (code, score) in enumerate(rows):
        # 프로그램 코드를 임시 파일에 저장 후 실행해 아이디어 dict 추출
        tmp = Path(results_dir) / f"_tmp_top{i}.py"
        tmp.write_text(code)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("top", tmp)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            idea = m.generate_idea()
            idea["combined_score"] = score
            out_path = out_dir / f"top{i+1}_score{score:.3f}.json"
            out_path.write_text(json.dumps(idea, ensure_ascii=False, indent=2))
            print(f"  [top{i+1}] score={score:.4f} → {out_path}")
        except Exception as e:
            print(f"  [warn] top{i+1} export 실패: {e}")
        finally:
            tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--islands", type=int, default=3)
    parser.add_argument("--max_eval_jobs", type=int, default=2)
    parser.add_argument("--results_dir", default="results/phase1")
    args = parser.parse_args()

    run(
        generations=args.generations,
        islands=args.islands,
        max_eval_jobs=args.max_eval_jobs,
        results_dir=args.results_dir,
    )
```

- [ ] **Step 2: 구문 검사**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -c "import ast; ast.parse(open('evolve/phase1_idea/run_phase1.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: 임포트 확인 (실행 없이)**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -c "
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch.scheduler import LocalJobConfig
print('All imports OK')
"
```

Expected: `All imports OK`

- [ ] **Step 4: 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add evolve/phase1_idea/run_phase1.py
git commit -m "feat(phase1): add ShinkaEvolveRunner script for idea evolution"
```

---

## Task 5: Phase 1 — 스모크 테스트 (3세대)

**Files:**
- No new files — 기존 코드 검증

- [ ] **Step 1: `ANTHROPIC_API_KEY` 환경변수 확인**

```bash
echo ${ANTHROPIC_API_KEY:0:5}...
```

Expected: `sk-an...` (설정되어 있어야 함). 없으면:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

- [ ] **Step 2: `--skip_llm` 플래그로 단축 진화 실행 (API 비용 없음)**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
# eval_idea.py 가 --skip_llm 을 받도록 eval 스크립트를 직접 호출 테스트
python3 evolve/phase1_idea/eval_idea.py \
    --program_path evolve/phase1_idea/seed/initial.py \
    --results_dir /tmp/test_phase1_eval \
    --skip_llm
```

Expected 출력:
```
[eval_idea] combined_score=0.xxxx | {'combined_score': ...}
```

- [ ] **Step 3: metrics.json 내용 확인**

```bash
cat /tmp/test_phase1_eval/metrics.json
```

Expected:
```json
{
  "combined_score": 0.5,
  "novelty": ...,
  "feasibility": 0.5,
  "impact": 0.5
}
```

- [ ] **Step 4: Phase 1 단위 테스트 전체 통과 확인**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -m pytest tests/test_phase1_evaluator.py -v
```

Expected: `3 passed`

---

## Task 6: Phase 2 — 아이디어→runfile 스캐폴드

**Files:**
- Create: `evolve/phase2_code/scaffold.py`

best idea JSON을 입력받아 pandapower 기반 `run_experiment()` runfile 초안을 LLM으로 생성한다.

- [ ] **Step 1: scaffold.py 작성**

```python
# evolve/phase2_code/scaffold.py
"""Phase 2 스캐폴드: best idea JSON → 초기 runfile.py 생성.

사용법:
  python evolve/phase2_code/scaffold.py \
      --idea evolved_ideas/top1_score0.72.json \
      --output evolve/phase2_code/generated/initial.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent.parent

SCAFFOLD_PROMPT_TEMPLATE = """당신은 에너지 시스템 연구 코드 전문가입니다.
아래 연구 아이디어를 구현하는 Python runfile을 작성하세요.

연구 아이디어:
{idea_json}

사용 가능한 도구:
{tools_context}

요구사항:
1. `run_experiment()` 함수를 정의하고, dict를 반환하세요.
   반환 dict에는 반드시 `"combined_score": float` 키가 있어야 합니다.
   (점수가 높을수록 좋음. 예: 라인 과부하율 감소 → 점수 높음)
2. ShinkaEvolve EVOLVE-BLOCK 마커를 핵심 알고리즘 부분에 삽입하세요:
   - `# === EVOLVE-BLOCK START: network_setup ===` / `# === EVOLVE-BLOCK END ===`
   - `# === EVOLVE-BLOCK START: optimization_strategy ===` / `# === EVOLVE-BLOCK END ===`
   - `# === EVOLVE-BLOCK START: metric_computation ===` / `# === EVOLVE-BLOCK END ===`
3. pandapower, pandapipes, pypsa, pyomo, cvxpy 중 적합한 라이브러리를 사용하세요.
4. 실행 시간은 60초 이내로 제한하세요.
5. import 문은 EVOLVE-BLOCK 밖 상단에 작성하세요.
6. 코드만 출력하고 설명은 생략하세요.

runfile.py 코드:
"""


def _load_tools_context() -> str:
    tools_dir = ROOT / "tools"
    texts = []
    for md in sorted(tools_dir.glob("*.md")):
        if md.name != "README.md":
            texts.append(f"=== {md.stem} ===\n{md.read_text()[:800]}")
    return "\n\n".join(texts) if texts else "(도구 정보 없음)"


def scaffold(idea_path: str, output_path: str) -> None:
    idea = json.loads(Path(idea_path).read_text())
    tools_context = _load_tools_context()

    prompt = SCAFFOLD_PROMPT_TEMPLATE.format(
        idea_json=json.dumps(idea, ensure_ascii=False, indent=2),
        tools_context=tools_context,
    )

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    code = msg.content[0].text.strip()

    # 코드블록 마크다운 제거
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(code)
    print(f"[scaffold] runfile 생성 완료: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea", required=True, help="아이디어 JSON 경로")
    parser.add_argument("--output", required=True, help="생성할 runfile.py 경로")
    args = parser.parse_args()
    scaffold(args.idea, args.output)
```

- [ ] **Step 2: 구문 검사**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -c "import ast; ast.parse(open('evolve/phase2_code/scaffold.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: tools/ 디렉터리에 pandapower.md 작성**

```markdown
<!-- tools/pandapower.md -->
# pandapower

AC/DC 전력망 시뮬레이션 라이브러리.

## 빠른 시작

```python
import pandapower as pp
import pandapower.networks as pn

net = pn.case30()          # IEEE 30-bus 테스트 네트워크
pp.runpp(net)              # AC 조류 계산 실행
loading = net.res_line["loading_percent"].max()  # 최대 라인 부하율
loss = net.res_line["pl_mw"].sum()               # 총 손실 (MW)
```

## 주요 API

- `pn.case14()`, `pn.case30()`, `pn.case118()` — 표준 테스트 네트워크
- `pp.runpp(net)` — AC 조류 계산
- `pp.rundcpp(net)` — DC 조류 계산
- `net.res_bus` — 모선 결과 (vm_pu, va_degree)
- `net.res_line` — 선로 결과 (loading_percent, pl_mw)
- `net.res_gen` — 발전기 결과

## combined_score 예시

```python
# 라인 과부하율 최소화 (낮을수록 좋으므로 음수로 변환)
score = float(-net.res_line["loading_percent"].max() / 100.0)
# 또는 전압 편차 최소화
score = float(-((net.res_bus["vm_pu"] - 1.0).abs().mean()))
```
```

- [ ] **Step 4: tools/README.md 작성**

```markdown
<!-- tools/README.md -->
# 에너지 시뮬레이션 툴 가이드

| 툴 | 용도 | 파일 |
|---|---|---|
| pandapower | AC/DC 전력망 시뮬레이션, 조류 계산 | pandapower.md |
| pypsa | 재생에너지 통합, 장기 에너지 최적화 | (추가 예정) |
| pandapipes | 열·가스 네트워크 시뮬레이션 | (추가 예정) |
| pyomo | 수리 최적화 (HiGHS 연동) | (추가 예정) |
| cvxpy | 볼록 최적화 | (추가 예정) |

## 선택 가이드

- **전력망 운영 최적화 (OPF)** → pandapower
- **재생에너지 비율 최적화** → pypsa
- **지역 냉난방 (DHC)** → pandapipes
- **혼합정수 최적화** → pyomo + HiGHS
- **볼록 완화** → cvxpy
```

- [ ] **Step 5: 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add evolve/phase2_code/scaffold.py tools/pandapower.md tools/README.md
git commit -m "feat(phase2): add scaffold.py and tools documentation"
```

---

## Task 7: Phase 2 — runfile 평가 스크립트

**Files:**
- Create: `evolve/phase2_code/eval_code.py`
- Test: `tests/test_phase2_evaluator.py`

- [ ] **Step 1: 테스트 먼저 작성**

```python
# tests/test_phase2_evaluator.py
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DUMMY_RUNFILE = '''
import pandapower as pp
import pandapower.networks as pn

def run_experiment():
    net = pn.case14()
    pp.runpp(net)
    score = float(-net.res_line["loading_percent"].max() / 100.0)
    return {"combined_score": score, "max_loading_pct": float(net.res_line["loading_percent"].max())}
'''

BROKEN_RUNFILE = '''
def run_experiment():
    raise RuntimeError("simulated failure")
'''

MISSING_SCORE_RUNFILE = '''
def run_experiment():
    return {"result": 42}  # combined_score 키 없음
'''


def _write_runfile(tmp_dir: str, content: str) -> str:
    path = os.path.join(tmp_dir, "runfile.py")
    Path(path).write_text(content)
    return path


def test_eval_writes_metrics_json():
    with tempfile.TemporaryDirectory() as tmp:
        prog = _write_runfile(tmp, DUMMY_RUNFILE)
        results_dir = os.path.join(tmp, "results")
        result = subprocess.run(
            ["python3", "evolve/phase2_code/eval_code.py",
             "--program_path", prog, "--results_dir", results_dir],
            capture_output=True, text=True,
            cwd="/home/syublee/AI-Scientist-Evolve-v1",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        metrics = json.loads(Path(results_dir, "metrics.json").read_text())
        assert "combined_score" in metrics
        assert isinstance(metrics["combined_score"], float)
        assert metrics["combined_score"] < 0  # 음수 (부하율 음수 변환)


def test_eval_handles_runtime_error():
    """실행 실패 시 combined_score=0.0 으로 fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        prog = _write_runfile(tmp, BROKEN_RUNFILE)
        results_dir = os.path.join(tmp, "results")
        result = subprocess.run(
            ["python3", "evolve/phase2_code/eval_code.py",
             "--program_path", prog, "--results_dir", results_dir],
            capture_output=True, text=True,
            cwd="/home/syublee/AI-Scientist-Evolve-v1",
        )
        # 실패해도 metrics.json은 써야 함
        assert os.path.exists(os.path.join(results_dir, "metrics.json"))
        metrics = json.loads(Path(results_dir, "metrics.json").read_text())
        assert metrics["combined_score"] == 0.0


def test_eval_handles_missing_score_key():
    """combined_score 키 없는 반환에도 fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        prog = _write_runfile(tmp, MISSING_SCORE_RUNFILE)
        results_dir = os.path.join(tmp, "results")
        subprocess.run(
            ["python3", "evolve/phase2_code/eval_code.py",
             "--program_path", prog, "--results_dir", results_dir],
            capture_output=True, text=True,
            cwd="/home/syublee/AI-Scientist-Evolve-v1",
        )
        metrics = json.loads(Path(results_dir, "metrics.json").read_text())
        assert metrics["combined_score"] == 0.0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -m pytest tests/test_phase2_evaluator.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` 또는 `FileNotFoundError`

- [ ] **Step 3: eval_code.py 구현**

```python
# evolve/phase2_code/eval_code.py
"""Phase 2 runfile 평가 스크립트.

ShinkaEvolve LocalJobConfig 가 subprocess로 호출:
  python eval_code.py --program_path <path> --results_dir <dir>

출력: results_dir/metrics.json  {"combined_score": float, ...}
실행 실패 시 combined_score=0.0 으로 fallback.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import signal
import sys
from pathlib import Path

TIMEOUT_SECONDS = 120  # 단일 실험 최대 실행 시간


def _load_and_run(program_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("runfile", program_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    if not hasattr(m, "run_experiment"):
        raise AttributeError(f"run_experiment() not found in {program_path}")

    result = m.run_experiment()

    if not isinstance(result, dict):
        raise TypeError(f"run_experiment() must return dict, got {type(result)}")

    if "combined_score" not in result:
        raise KeyError("combined_score key missing from run_experiment() return value")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # SIGALRM으로 타임아웃 (Linux only)
        def _timeout_handler(signum, frame):
            raise TimeoutError(f"run_experiment() exceeded {TIMEOUT_SECONDS}s")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)

        result = _load_and_run(args.program_path)
        signal.alarm(0)

        metrics = {"combined_score": float(result["combined_score"]), **{
            k: v for k, v in result.items() if k != "combined_score"
        }}
        print(f"[eval_code] combined_score={metrics['combined_score']:.4f}")

    except Exception as e:
        signal.alarm(0)
        print(f"[eval_code] ERROR: {e}", file=sys.stderr)
        metrics = {"combined_score": 0.0, "error": str(e)}

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -m pytest tests/test_phase2_evaluator.py -v
```

Expected: `3 passed`

- [ ] **Step 5: 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add evolve/phase2_code/eval_code.py tests/test_phase2_evaluator.py
git commit -m "feat(phase2): add code evaluator with timeout and error fallback"
```

---

## Task 8: Phase 2 — ShinkaEvolveRunner 실행 스크립트

**Files:**
- Create: `evolve/phase2_code/run_phase2.py`

- [ ] **Step 1: run_phase2.py 작성**

```python
# evolve/phase2_code/run_phase2.py
"""Phase 2: 실험 코드 진화 실행.

사용법:
  python evolve/phase2_code/run_phase2.py \
      --idea evolved_ideas/top1_score0.72.json \
      --generations 15 --max_eval_jobs 2

결과: results/phase2/<exp_name>/ 에 SQLite DB + best runfile 저장
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import shinka
from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch.scheduler import LocalJobConfig

ROOT = Path(__file__).parent.parent.parent
EVAL_SCRIPT = str(Path(__file__).parent / "eval_code.py")

TASK_SYS_MSG_TEMPLATE = """당신은 에너지 시스템 실험 코드 전문가입니다.
아래 연구 아이디어를 구현하는 Python runfile의 EVOLVE-BLOCK을 개선하세요.

연구 아이디어:
{idea_summary}

규칙:
1. EVOLVE-BLOCK 마커 외부는 절대 수정하지 마세요.
2. run_experiment() 함수 시그니처를 유지하세요.
3. 반환 dict에 반드시 "combined_score": float 키가 있어야 합니다.
4. pandapower, pandapipes, pypsa, pyomo, cvxpy 라이브러리를 활용하세요.
5. 실행 시간은 120초 이내로 제한하세요.
6. combined_score가 높을수록 좋은 결과입니다.
"""


def _scaffold_if_needed(idea_path: str, exp_dir: str) -> str:
    """초기 runfile이 없으면 scaffold.py로 생성."""
    initial_path = os.path.join(exp_dir, "initial.py")
    if os.path.exists(initial_path):
        return initial_path

    from evolve.phase2_code.scaffold import scaffold
    scaffold(idea_path=idea_path, output_path=initial_path)
    return initial_path


def run(idea_path: str, generations: int, max_eval_jobs: int, results_dir: str) -> None:
    idea = json.loads(Path(idea_path).read_text())
    exp_name = Path(idea_path).stem.replace(" ", "_")
    exp_dir = os.path.join(results_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    initial_py = _scaffold_if_needed(idea_path, exp_dir)

    idea_summary = (
        f"제목: {idea.get('Title', '(없음)')}\n"
        f"문제: {idea.get('Problem', '')[:200]}\n"
        f"방법: {idea.get('Method', '')[:200]}"
    )

    evo_config = EvolutionConfig(
        task_sys_msg=TASK_SYS_MSG_TEMPLATE.format(idea_summary=idea_summary),
        language="python",
        num_generations=generations,
        patch_types=["diff", "full"],
        patch_type_probs=[0.7, 0.3],
        llm_models=["claude-sonnet-4-6"],
        llm_kwargs={"temperatures": [0.7, 1.0], "max_tokens": 8192},
        embedding_model=None,
        init_program_path=initial_py,
        results_dir=exp_dir,
        meta_rec_interval=5,
    )

    db_config = DatabaseConfig(
        db_path=os.path.join(exp_dir, "evolution_db.sqlite"),
        num_islands=2,
        archive_size=20,
        exploitation_ratio=0.3,
        elite_selection_ratio=0.3,
    )

    job_config = LocalJobConfig(
        eval_program_path=EVAL_SCRIPT,
        conda_env="ai_scientist_energy",
    )

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=max_eval_jobs,
        max_proposal_jobs=max_eval_jobs + 1,
        max_db_workers=2,
        verbose=True,
    )
    runner.run()

    print(f"\n[Phase 2 완료] 결과: {exp_dir}")
    _export_best_runfile(exp_dir, exp_name)


def _export_best_runfile(exp_dir: str, exp_name: str) -> None:
    """DB에서 best runfile을 workspaces/<exp_name>/best_runfile.py 에 저장."""
    import sqlite3
    db_path = os.path.join(exp_dir, "evolution_db.sqlite")
    if not os.path.exists(db_path):
        return

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT program_str, combined_score FROM programs ORDER BY combined_score DESC LIMIT 1"
    ).fetchone()
    conn.close()

    if not row:
        print("[warn] DB에 결과 없음")
        return

    code, score = row
    out_dir = ROOT / "workspaces" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_runfile.py"
    best_path.write_text(code)
    print(f"  [best] score={score:.4f} → {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea", required=True, help="아이디어 JSON 경로")
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--max_eval_jobs", type=int, default=2)
    parser.add_argument("--results_dir", default="results/phase2")
    args = parser.parse_args()

    run(
        idea_path=args.idea,
        generations=args.generations,
        max_eval_jobs=args.max_eval_jobs,
        results_dir=args.results_dir,
    )
```

- [ ] **Step 2: 구문 검사**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -c "import ast; ast.parse(open('evolve/phase2_code/run_phase2.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

- [ ] **Step 3: 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add evolve/phase2_code/run_phase2.py
git commit -m "feat(phase2): add ShinkaEvolveRunner script for code evolution"
```

---

## Task 9: Phase 2 — 스모크 테스트 (case14, 3세대)

**Files:**
- Create: `evolve/phase2_code/generated/initial.py` (테스트용 seed runfile)

- [ ] **Step 1: 테스트용 초기 runfile 수동 작성**

```python
# evolve/phase2_code/generated/initial.py
"""Phase 2 테스트용 초기 runfile (scaffold 없이 수동 작성)."""
import pandapower as pp
import pandapower.networks as pn


def run_experiment() -> dict:
    # === EVOLVE-BLOCK START: network_setup ===
    net = pn.case14()
    # === EVOLVE-BLOCK END ===

    # === EVOLVE-BLOCK START: optimization_strategy ===
    pp.runpp(net)
    # === EVOLVE-BLOCK END ===

    # === EVOLVE-BLOCK START: metric_computation ===
    max_loading = float(net.res_line["loading_percent"].max())
    score = float(-max_loading / 100.0)
    # === EVOLVE-BLOCK END ===

    return {
        "combined_score": score,
        "max_loading_pct": max_loading,
        "total_loss_mw": float(net.res_line["pl_mw"].sum()),
    }
```

- [ ] **Step 2: eval_code.py 직접 실행 확인**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 evolve/phase2_code/eval_code.py \
    --program_path evolve/phase2_code/generated/initial.py \
    --results_dir /tmp/test_phase2_eval
cat /tmp/test_phase2_eval/metrics.json
```

Expected: `{"combined_score": -0.xxxx, "max_loading_pct": xx.x, "total_loss_mw": x.x}`

- [ ] **Step 3: Phase 2 테스트 전체 통과 확인**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -m pytest tests/test_phase2_evaluator.py -v
```

Expected: `3 passed`

- [ ] **Step 4: 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add evolve/phase2_code/generated/initial.py
git commit -m "test(phase2): add manual seed runfile for smoke testing"
```

---

## Task 10: references/ 스텁 작성

**Files:**
- Create: `references/README.md`
- Create: `references/papers/.gitkeep`
- Create: `references/datasets/.gitkeep`

- [ ] **Step 1: references/README.md 작성**

```markdown
<!-- references/README.md -->
# References 폴더 사용 가이드

Phase 1 아이디어 평가 시 이 폴더의 내용이 LLM 채점 컨텍스트로 주입됩니다.

## 폴더 구조

```
references/
├── papers/          # 논문 PDF 또는 .md 요약 파일
│   └── example.md  # 예: "논문 제목\n\n핵심 내용 요약"
└── datasets/        # 데이터셋 메타정보 또는 샘플 CSV
    └── example.md  # 예: "데이터셋 이름, 크기, 특징, 출처"
```

## 사용 방법

1. `papers/` 에 논문 요약 .md 파일을 추가하면 Feasibility·Impact 채점에 활용됩니다.
2. `datasets/` 에 데이터셋 메타정보를 추가하면 "이 데이터로 실험 가능한가"를 판단하는 데 활용됩니다.
3. 파일당 1000자 이내로 작성하면 토큰 효율이 좋습니다.
4. 파일을 추가한 후 Phase 1을 재실행하면 자동으로 반영됩니다.

## 예시 (papers/vortex_r2_summary.md)

```
# Vortex-R2: Relationship-Regularized Tabular-to-Image for IIoT

키워드: IIoT, fault diagnosis, tabular-to-image, label scarcity

핵심 기여:
- 반상관 센서를 이미지 중심에 배치하는 Vortex-R2 레이아웃
- 1% 라벨 환경에서 ResNet-18 과적합 억제
- SECOM, CWRU, UCI HAR 데이터셋에서 기존 대비 3% 향상
```
```

- [ ] **Step 2: 빈 디렉터리 유지용 .gitkeep 생성 및 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
touch references/papers/.gitkeep references/datasets/.gitkeep
git add references/README.md references/papers/.gitkeep references/datasets/.gitkeep tools/pandapower.md tools/README.md
git commit -m "docs: add references and tools documentation stubs"
```

---

## Task 11: pipeline.py 오케스트레이터

**Files:**
- Create: `pipeline.py`

- [ ] **Step 1: pipeline.py 작성**

```python
# pipeline.py
"""AI-Scientist-Evolve-v1 파이프라인 오케스트레이터.

사용법:
  python pipeline.py \
      --seed evolve/phase1_idea/seed/initial.py \
      --phase1_generations 20 \
      --phase2_generations 15 \
      --journal energies \
      --top_k 5

  # Phase 2부터 재시작 (Phase 1 결과 재활용):
  python pipeline.py --start_phase 2 --journal energies

옵션:
  --start_phase N   1(기본) | 2 | 3  — 해당 Phase부터 시작
  --top_k N         Phase 1 상위 N개를 Phase 2로 전달 (기본 5)
  --skip_writeup    Phase 3 논문 작성 생략
  --dry_run         실제 실행 없이 설정 출력만
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def _run_phase1(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("PHASE 1: 아이디어 진화")
    print("=" * 60)

    cmd = [
        sys.executable,
        str(ROOT / "evolve/phase1_idea/run_phase1.py"),
        "--generations", str(args.phase1_generations),
        "--islands", "3",
        "--max_eval_jobs", "2",
        "--results_dir", "results/phase1",
    ]
    if args.dry_run:
        print(f"[dry_run] {' '.join(cmd)}")
        return
    subprocess.run(cmd, check=True, cwd=ROOT)


def _run_phase2(args: argparse.Namespace) -> None:
    print("\n" + "=" * 60)
    print("PHASE 2: 실험 코드 진화")
    print("=" * 60)

    idea_files = sorted(glob.glob(str(ROOT / "evolved_ideas/*.json")))
    if not idea_files:
        raise FileNotFoundError(
            "evolved_ideas/ 가 비어 있습니다. Phase 1을 먼저 실행하세요."
        )

    top_ideas = idea_files[: args.top_k]
    print(f"[phase2] {len(top_ideas)}개 아이디어에 대해 코드 진화 실행")

    for idea_path in top_ideas:
        print(f"\n  ▸ {Path(idea_path).name}")
        cmd = [
            sys.executable,
            str(ROOT / "evolve/phase2_code/run_phase2.py"),
            "--idea", idea_path,
            "--generations", str(args.phase2_generations),
            "--max_eval_jobs", "2",
            "--results_dir", "results/phase2",
        ]
        if args.dry_run:
            print(f"  [dry_run] {' '.join(cmd)}")
            continue
        subprocess.run(cmd, check=True, cwd=ROOT)


def _run_phase3(args: argparse.Namespace) -> None:
    if args.skip_writeup:
        print("\n[phase3] --skip_writeup 지정, 논문 생성 생략")
        return

    print("\n" + "=" * 60)
    print("PHASE 3: 논문 생성 (AI-Scientist-v2 재활용)")
    print("=" * 60)

    best_runfiles = sorted(glob.glob(str(ROOT / "workspaces/*/best_runfile.py")))
    if not best_runfiles:
        raise FileNotFoundError("workspaces/ 에 best_runfile.py 없음. Phase 2를 먼저 실행하세요.")

    # AI-Scientist-v2 perform_mdpi_writeup.py 직접 호출
    writeup_script = ROOT / "ai_scientist/perform_mdpi_writeup.py"
    if not writeup_script.exists():
        raise FileNotFoundError(f"{writeup_script} 없음. ai_scientist 심볼릭 링크를 확인하세요.")

    for runfile in best_runfiles[:args.top_k]:
        exp_name = Path(runfile).parent.name
        working_dir = ROOT / "workspaces" / exp_name
        papers_dir = ROOT / "papers" / exp_name
        papers_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(writeup_script),
            "--experiment", str(working_dir),
            "--journal", args.journal,
        ]
        if args.dry_run:
            print(f"  [dry_run] {' '.join(cmd)}")
            continue
        print(f"\n  ▸ {exp_name}")
        subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="evolve/phase1_idea/seed/initial.py")
    parser.add_argument("--phase1_generations", type=int, default=20)
    parser.add_argument("--phase2_generations", type=int, default=15)
    parser.add_argument("--journal", default="energies")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--start_phase", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--skip_writeup", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.start_phase <= 1:
        _run_phase1(args)
    if args.start_phase <= 2:
        _run_phase2(args)
    if args.start_phase <= 3:
        _run_phase3(args)

    print("\n✓ 파이프라인 완료")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: dry_run으로 동작 확인**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
# evolved_ideas 가 없어도 --start_phase 2 는 오류; phase1만 dry_run
python3 pipeline.py --phase1_generations 3 --phase2_generations 3 --dry_run --start_phase 1
```

Expected: `[dry_run] python3 evolve/phase1_idea/run_phase1.py ...`

- [ ] **Step 3: 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add pipeline.py
git commit -m "feat: add pipeline.py orchestrating Phase 1→2→3"
```

---

## Task 12: 통합 테스트 (단축 설정)

**Files:**
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: 통합 테스트 작성**

```python
# tests/test_pipeline.py
"""통합 테스트: pipeline.py --dry_run 실행 확인."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def test_pipeline_dry_run_phase1():
    """Phase 1 dry_run이 오류 없이 설정을 출력하는지 확인."""
    result = subprocess.run(
        [sys.executable, "pipeline.py",
         "--phase1_generations", "3",
         "--dry_run",
         "--start_phase", "1"],
        capture_output=True, text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "dry_run" in result.stdout
    assert "run_phase1.py" in result.stdout


def test_all_modules_importable():
    """핵심 모듈들이 import 오류 없이 로드되는지 확인."""
    result = subprocess.run(
        [sys.executable, "-c", """
from evolve.phase1_idea.eval_idea import score_novelty
from evolve.phase1_idea.seed.initial import generate_idea
from evolve.phase2_code.eval_code import _load_and_run
from evolve.phase2_code.scaffold import scaffold
print("All imports OK")
"""],
        capture_output=True, text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "All imports OK" in result.stdout


def test_eval_idea_smoke():
    """eval_idea.py --skip_llm 스모크 테스트."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "evolve/phase1_idea/eval_idea.py",
             "--program_path", "evolve/phase1_idea/seed/initial.py",
             "--results_dir", os.path.join(tmp, "results"),
             "--skip_llm"],
            capture_output=True, text=True,
            cwd=str(ROOT),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "combined_score" in result.stdout
```

- [ ] **Step 2: 테스트 실패 확인 (pipeline.py 없으면)**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -m pytest tests/test_pipeline.py -v 2>&1 | head -20
```

- [ ] **Step 3: 전체 테스트 스위트 실행**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
python3 -m pytest tests/ -v
```

Expected:
```
tests/test_phase1_evaluator.py::test_score_novelty_returns_float PASSED
tests/test_phase1_evaluator.py::test_score_feasibility_returns_float PASSED
tests/test_phase1_evaluator.py::test_eval_writes_metrics_json PASSED
tests/test_phase2_evaluator.py::test_eval_writes_metrics_json PASSED
tests/test_phase2_evaluator.py::test_eval_handles_runtime_error PASSED
tests/test_phase2_evaluator.py::test_eval_handles_missing_score_key PASSED
tests/test_pipeline.py::test_pipeline_dry_run_phase1 PASSED
tests/test_pipeline.py::test_all_modules_importable PASSED
tests/test_pipeline.py::test_eval_idea_smoke PASSED
9 passed
```

- [ ] **Step 4: 최종 커밋**

```bash
cd /home/syublee/AI-Scientist-Evolve-v1
git add tests/test_pipeline.py
git commit -m "test: add integration tests for pipeline dry_run and module imports"
```

---

## Self-Review

**1. Spec coverage:**

| 스펙 항목 | 커버 태스크 |
|---|---|
| Phase 1 아이디어 진화 | Task 2, 3, 4, 5 |
| 멀티아일랜드 3개 | Task 4 (`num_islands=3`) |
| 20세대, 상위 5개 전달 | Task 4, 11 |
| Phase 1 채점 (Novelty 0.35 + Feasibility 0.35 + Impact 0.30) | Task 3 |
| references/ 컨텍스트 주입 | Task 3 (`_load_references`) |
| Phase 2 scaffold | Task 6 |
| Phase 2 코드 진화 | Task 7, 8 |
| Phase 2 타임아웃 120초 | Task 7 (`TIMEOUT_SECONDS=120`) |
| Phase 3 AI-Scientist-v2 재활용 | Task 11 (`perform_mdpi_writeup.py` 호출) |
| pipeline.py 오케스트레이션 | Task 11 |
| `--start_phase` 옵션 | Task 11 |
| tools/ 폴더 활용 | Task 6, Task 10 |
| 단위 테스트 | Task 3, 5, 7, 9 |
| 통합 테스트 | Task 12 |
| environment.yml | Task 1 |
| 심볼릭 링크 ai_scientist/, data/ | Task 1 |

**2. Placeholder scan:** 없음 — 모든 코드 블록이 완전히 작성됨.

**3. Type consistency:**
- `generate_idea()` → `dict` (Task 2, 3에서 일관)
- `score_novelty(idea, seed_dir)` → `float` (Task 3에서 정의, 테스트에서 호출)
- `_call_llm_score(prompt)` → `float` (Task 3에서 정의, monkeypatch 대상)
- `run_experiment()` → `dict` with `combined_score` key (Task 7, 8, 9에서 일관)
- `scaffold(idea_path, output_path)` → `None` (Task 6, 8에서 일관)
