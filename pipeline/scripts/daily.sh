#!/usr/bin/env bash
# Daily paper-trading step: pull new bars, process unprocessed trading days, rewrite the Phase 7 report.
# Run after the US close, e.g. crontab: 30 18 * * 1-5 bash /path/to/quantico/pipeline/scripts/daily.sh
set -euo pipefail
cd "$(dirname "$0")/.."
make -C ../simulator
conda run -n quantico --no-capture-output python src/paper.py run >> paper.log 2>&1
