#!/usr/bin/env bash
# Removes the launchd schedule installed by install_schedule.sh.
set -euo pipefail
PLIST="$HOME/Library/LaunchAgents/com.quantico.daily.plist"
launchctl unload "$PLIST" 2>/dev/null || true
rm -f "$PLIST"
echo "removed: $PLIST"
