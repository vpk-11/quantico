#!/usr/bin/env bash
# Installs the launchd schedule: weekdays 17:30 local time, plus RunAtLoad so a sleeping laptop
# catches up on wake (daily.sh/paper.py run processes every missed trading day in order).
# This script is not run automatically by any tool in this repo; run it yourself when ready.
set -euo pipefail
PIPELINE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/com.quantico.daily.plist"
mkdir -p "$PIPELINE_DIR/logs"
sed "s|__PIPELINE_DIR__|$PIPELINE_DIR|g" "$PIPELINE_DIR/scripts/launchd/com.quantico.daily.plist.template" > "$PLIST"
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"
echo "installed: $PLIST"
echo "17:30 local time assumed to be US/Eastern or close enough; adjust the plist if not."
