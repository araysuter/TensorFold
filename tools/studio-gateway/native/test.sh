#!/bin/bash
set -euo pipefail
SOURCE=${1:?source checkout directory}
ZIG=${2:?Zig 0.17.0 executable}
cd "$SOURCE"
"$ZIG" test --dep sys --dep lanes -Mroot=zig/src/core/session_file.zig \
 -Msys=zig/src/platform/sys.zig --dep sys -Mlanes=zig/src/core/lanes/lanes.zig \
 -lc
