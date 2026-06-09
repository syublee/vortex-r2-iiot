#!/usr/bin/env bash
# =============================================================================
# run_scientist.sh  (AI Scientist — domain-agnostic)
#
# Claude 구독(claude-code/opus, API 키 없음) 환경에서 AI Scientist 를
# 토픽 .md 파일 하나로 한 번에 실행한다:
#   preflight 점검  ->  ideation  ->  실험(BFTS) + MDPI writeup + review
#
# 사용법:
#   ./run_scientist.sh <topic.md> [옵션]
#
# 옵션:
#   --max-gen N         ideation 제안 생성 수            (기본 5)
#   --reflections M     제안당 reflection 라운드 수      (기본 3)
#   --journal J         대상 저널명 (기본 energies)
#                       예: energies, sustainability, nature, ieee-access, plos-one
#   --page-limit P      MDPI soft page-limit             (기본 15)
#   --num-cite R        citation 라운드 수               (기본 20)
#   --attempt-id K      동일 idea 의 attempt 구분용      (기본 0)
#   --idea-idx K        실행할 아이디어 인덱스           (기본 0)
#   --skip-ideation     이미 만들어진 <topic>.json 사용
#   --skip-writeup      논문 작성 단계 생략
#   --skip-review       리뷰 단계 생략
#   --bfts-config F     BFTS config yaml 경로 (기본 bfts_config_general.yaml)
#
# 환경변수:
#   AI_SCIENTIST_MODEL       사용할 모델  (기본 claude-code/opus)
#   AI_SCIENTIST_MAX_TURNS   Claude Code 한 호출 당 최대 턴 수 (기본 24)
#
# 권장: 수 시간 걸리므로 로그를 정규 파일로 리다이렉트하세요:
#   nohup ./run_scientist.sh ai_scientist/ideas/my_topic.md \
#       --journal nature > run.log 2>&1 &
#   tail -f run.log
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- 기본값 / 인자 파싱 -------------------------------------------------------
MODEL="${AI_SCIENTIST_MODEL:-claude-code/opus}"
MAX_GEN=5
REFL=3
JOURNAL="energies"
PAGE_LIMIT=15
NUM_CITE=20
ATTEMPT=0
IDEA_IDX=0
SKIP_IDEATION=0
SKIP_WRITEUP=0
SKIP_REVIEW=0
SKIP_SKILLOPT=0
BFTS_CONFIG="bfts_config_general.yaml"
TOPIC_MD=""

usage() {
  awk 'NR>=2 && /^#/ {sub(/^# ?/,""); print; next} NR>=2 {exit}' "$0"
}

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
    --skip-skillopt) SKIP_SKILLOPT=1; shift ;;
    --bfts-config)   BFTS_CONFIG="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    -*)              echo "알 수 없는 옵션: $1" >&2; usage; exit 2 ;;
    *)               TOPIC_MD="$1"; shift ;;
  esac
done

err()  { echo -e "\033[31m[오류]\033[0m $*" >&2; }
warn() { echo -e "\033[33m[경고]\033[0m $*" >&2; }
info() { echo -e "\033[36m[진행]\033[0m $*"; }

# --- Preflight ---------------------------------------------------------------
info "Preflight 점검 ..."

if [[ -z "$TOPIC_MD" ]]; then
  err "토픽 .md 파일을 인자로 주세요."
  usage; exit 2
fi
if [[ "$TOPIC_MD" != *.md ]]; then
  err "토픽 파일은 .md 여야 합니다: $TOPIC_MD"; exit 2
fi
if [[ ! -f "$TOPIC_MD" ]]; then
  err "토픽 파일을 찾을 수 없습니다: $TOPIC_MD"; exit 2
fi
if [[ ! -f "$BFTS_CONFIG" ]]; then
  err "BFTS config 파일을 찾을 수 없습니다: $BFTS_CONFIG"; exit 2
fi

if ! command -v claude >/dev/null 2>&1; then
  err "'claude' CLI 가 없습니다. 설치: npm install -g @anthropic-ai/claude-code"
  exit 1
fi

if [[ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" && ! -f "$HOME/.claude/.credentials.json" ]]; then
  err "Claude 구독 인증이 없습니다. 'claude setup-token' 실행하세요."
  exit 1
fi

if [[ -n "${ANTHROPIC_API_KEY:-}" || -n "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  warn "ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN 감지 → 이 실행에서 해제 후 구독으로 진행"
  unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN || true
fi

if ! python -c "import claude_agent_sdk" >/dev/null 2>&1; then
  err "claude_agent_sdk 가 없습니다. 'pip install -r requirements.txt' 실행하세요."
  exit 1
fi

# MDPI LaTeX class 점검
if [[ "$SKIP_WRITEUP" -eq 0 ]]; then
  if [[ ! -f "ai_scientist/blank_mdpi_latex/Definitions/mdpi.cls" ]]; then
    warn "MDPI LaTeX class 없음. ./setup_mdpi_template.sh 로 설치하세요."
  fi
  if ! command -v pdflatex >/dev/null 2>&1; then
    warn "pdflatex 없음. PDF 컴파일이 건너뛰어질 수 있습니다."
  fi
fi

IDEAS_JSON="${TOPIC_MD%.md}.json"
TOPIC_NAME="$(basename "${TOPIC_MD%.md}")"

echo "============================================================"
echo " 토픽       : $TOPIC_MD"
echo " 저널       : $JOURNAL"
echo " BFTS config: $BFTS_CONFIG"
echo " 아이디어   : $IDEAS_JSON"
echo "============================================================"

# --- Step 1: Ideation --------------------------------------------------------
if [[ "$SKIP_IDEATION" -eq 0 ]]; then
  info "Step 1/2 — Ideation 시작"
  python ai_scientist/perform_ideation_temp_free.py \
    --workshop-file "$TOPIC_MD" \
    --model "$MODEL" \
    --max-num-generations "$MAX_GEN" \
    --num-reflections "$REFL"
  if [[ ! -f "$IDEAS_JSON" ]]; then
    err "Ideation 후 $IDEAS_JSON 가 생성되지 않았습니다."
    exit 1
  fi
  info "Ideation 완료 → $IDEAS_JSON"
else
  info "Ideation 건너뜀 (--skip-ideation)"
  if [[ ! -f "$IDEAS_JSON" ]]; then
    err "$IDEAS_JSON 가 없습니다. Ideation을 먼저 실행하거나 파일을 확인하세요."
    exit 1
  fi
fi

# --- Step 2: 실험 + Writeup + Review -----------------------------------------
info "Step 2/2 — 실험(BFTS) + Writeup + Review 시작"

LAUNCH_ARGS=(
  --load_ideas "$IDEAS_JSON"
  --bfts_config_path "$BFTS_CONFIG"
  --model_writeup "$MODEL"
  --model_writeup_small "$MODEL"
  --model_citation "$MODEL"
  --model_review "$MODEL"
  --model_agg_plots "$MODEL"
  --writeup-type mdpi
  --journal "$JOURNAL"
  --page-limit "$PAGE_LIMIT"
  --num_cite_rounds "$NUM_CITE"
  --attempt_id "$ATTEMPT"
  --idea_idx "$IDEA_IDX"
)
[[ "$SKIP_WRITEUP" -eq 1 ]] && LAUNCH_ARGS+=( --skip_writeup )
[[ "$SKIP_REVIEW"  -eq 1 ]] && LAUNCH_ARGS+=( --skip_review )

# 시드 코드: ideas/<topic>.py 가 있으면 자동으로 --load_code 추가
IDEAS_PY="${IDEAS_JSON%.json}.py"
[[ -f "$IDEAS_PY" ]] && LAUNCH_ARGS+=( --load_code ) && info "시드 코드 발견 → --load_code 활성화: $IDEAS_PY"

python launch_scientist_bfts.py "${LAUNCH_ARGS[@]}"

LATEST="$(ls -dt experiments/*_attempt_"${ATTEMPT}"/ 2>/dev/null | head -1 || true)"
echo "============================================================"
if [[ -n "$LATEST" ]]; then
  info "완료. 결과 디렉터리: $LATEST"
  PDF=$(ls "$LATEST"*.pdf 2>/dev/null | head -1 || true)
  [[ -n "$PDF" ]] && info "논문 PDF: $PDF"
else
  warn "실험 결과 디렉터리를 찾을 수 없습니다."
fi

# --- SkillOpt: Stage Goals 최적화 ------------------------------------------
if [[ "${SKIP_SKILLOPT:-0}" -ne 1 && -n "$LATEST" ]]; then
  info "SkillOpt: Stage Goals 최적화 실행 → $LATEST"
  python skillopt/optimize_stage_goals.py --experiment-dir "$LATEST" \
      --model "$MODEL" \
    || warn "SkillOpt 실패 — agent_manager.py 미수정"
fi
echo "============================================================"
