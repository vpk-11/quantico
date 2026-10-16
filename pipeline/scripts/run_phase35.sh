#!/usr/bin/env bash
# Phase 3.5: run the candidates, then verify. Takes several hours (random forest nulls).
# Usage: bash pipeline/scripts/run_phase35.sh   (log: pipeline/phase35.log)
set -euo pipefail
cd "$(dirname "$0")/.."

baseline_hash() {
  sqlite3 data/research_store.db "SELECT printf('%s|%s|%s|%d|%.17g|%.17g', ticker, date, model_id, fold_year, prediction, target) FROM predictions WHERE model_id='ridge_h5_baseline' ORDER BY ticker, date" | shasum -a 256
}

{
  echo "baseline predictions hash before: $(baseline_hash)"
  echo "expected (pre-phase):             32ec0dd25700e19377d09a76beba66df178d2069f66bcb7c435c1b82dda46653  -"
  caffeinate -i conda run -n quantico --no-capture-output python -u src/escalation.py run
  caffeinate -i conda run -n quantico --no-capture-output python -u src/escalation.py verify
  echo "baseline predictions hash after:  $(baseline_hash)"
} 2>&1 | tee phase35.log
