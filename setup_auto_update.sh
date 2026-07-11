#!/bin/bash
# Set up weekly automatic model updates on macOS (every Sunday at 9:00 AM)
#
# Usage: ./setup_auto_update.sh
# Remove: ./setup_auto_update.sh --uninstall

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$(which python3)"
PLIST_NAME="com.mentalhealth.bioactivity.update"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/${PLIST_NAME}.plist"
LOG_DIR="$PROJECT_DIR/data/logs"

mkdir -p "$PLIST_DIR" "$LOG_DIR"

if [ "$1" = "--uninstall" ]; then
  launchctl unload "$PLIST_PATH" 2>/dev/null || true
  rm -f "$PLIST_PATH"
  echo "Removed auto-update schedule."
  exit 0
fi

cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${PLIST_NAME}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON}</string>
        <string>${PROJECT_DIR}/update_models.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${PROJECT_DIR}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/auto_update.log</string>
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/auto_update_error.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/${PLIST_NAME}" 2>/dev/null || launchctl unload "$PLIST_PATH" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH" 2>/dev/null || launchctl load "$PLIST_PATH"

echo "Auto-update installed."
echo "  Schedule: every Sunday at 9:00 AM"
echo "  Script:   ${PROJECT_DIR}/update_models.py"
echo "  Logs:     ${LOG_DIR}/auto_update.log"
echo ""
echo "Run now manually:  python update_models.py --force"
echo "Check status:      python update_models.py --status"
echo "Uninstall:         ./setup_auto_update.sh --uninstall"
