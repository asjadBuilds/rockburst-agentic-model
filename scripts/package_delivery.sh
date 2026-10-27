#!/usr/bin/env bash
# Build a clean client delivery zip (Round-3 remediation).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/dist/rockburst-agentic-ai-remediation.zip}"
NAME="rockburst-agentic-ai"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$(dirname "$OUT")" "$STAGE/$NAME"

rsync -a \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '.env' \
  --exclude '.cursor/' \
  --exclude '.pytest_cache/' \
  --exclude '__pycache__/' \
  --exclude '**/__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.log' \
  --exclude 'catboost_info/' \
  --exclude 'logs/' \
  --exclude 'dist/' \
  --exclude 'artifacts/planner_cache/' \
  --exclude 'artifacts/tuning_cache/' \
  --exclude 'artifacts/archive/' \
  --exclude 'artifacts/models/fusion_fold_tmp.joblib' \
  --exclude 'reports/claude_agent_context_pack.md' \
  --exclude 'Feedback.docx' \
  --exclude 'Round2 Feedback.docx' \
  --exclude 'Round3 Feedback.docx' \
  --exclude '.ipynb_checkpoints/' \
  --exclude 'requirements-unpinned.bak.txt' \
  --exclude '*.egg-info/' \
  --exclude 'build/' \
  --exclude '.cursorignore' \
  "$ROOT/" "$STAGE/$NAME/"

cat > "$STAGE/$NAME/README_DELIVERY.txt" << 'EOF'
Rockburst Agentic AI — client delivery (Round-3 remediation)

Start here (same order as DELIVERABLES.md / COVER_NOTE.md):
  1. COVER_NOTE.md
  2. reports/feedback_response.md   (feedback evidence map)
  3. DELIVERABLES.md                (brief §7/§8 → file map; canonical)
  4. reports/technical_report.md    (full-mode numbers only)
  5. reports/methodology.md
  6. artifacts/figures/             (F1–F17, captions.md)
  7. notebooks/end_to_end_demo.ipynb
     Execute: jupyter nbconvert --to notebook --execute notebooks/end_to_end_demo.ipynb --inplace

Regenerate examples/figures from shipped models:
  .venv/bin/python scripts/generate_examples.py
  .venv/bin/python scripts/generate_figures.py

Setup:
  python3.12 -m venv .venv
  .venv/bin/pip install -r requirements.txt
  .venv/bin/pip install -e .
  # Models under artifacts/models/ (sha256-pinned in notebook)

Excluded from this zip (by design):
  .env, .venv, .cursor/, catboost_info/, planner_cache/, tuning_cache/,
  reports/claude_agent_context_pack.md, feedback DOCX files, artifacts/archive/
EOF

python_bin="$ROOT/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python3"
fi
"$python_bin" "$STAGE/$NAME/scripts/scan_client_content.py" "$STAGE/$NAME"

(cd "$STAGE/$NAME" && find . -type f ! -name DELIVERY_MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > DELIVERY_MANIFEST.sha256)

rm -f "$OUT"
(cd "$STAGE" && zip -r -q "$OUT" "$NAME")
echo "Wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo "Sanity: forbidden paths (expect empty):"
unzip -l "$OUT" | grep -E '\.cursor/|Feedback\.docx|(^|/)\\.env$|claude_agent|catboost_info|planner_cache|\\.venv/' || true
