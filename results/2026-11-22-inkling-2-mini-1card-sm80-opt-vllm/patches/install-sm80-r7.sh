#!/usr/bin/env bash
# install-sm80-r7.sh [DEST_DIR] [BASE_INKLING2_DIR] — alias for install-sm80-r3.sh (installs Round-7 Best stack)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/install-sm80-r3.sh" "$@"
