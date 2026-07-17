#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

python src/preflight.py

python -m py_compile \
    src/phase1.py \
    src/phase2.py \
    src/phase3.py \
    src/deployment.py \
    src/run_project.py \
    src/preflight.py \
    dashboard/app.py

python -m pytest tests -v