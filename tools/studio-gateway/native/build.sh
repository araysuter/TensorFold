#!/bin/bash
# Reproducible native engine for the Studio gateway. Only builds; never starts or replaces a server.
set -euo pipefail
SOURCE=${1:?Usage: build.sh SOURCE_DIRECTORY ZIG_0_17_EXECUTABLE}
ZIG=${2:?Usage: build.sh SOURCE_DIRECTORY ZIG_0_17_EXECUTABLE}
PATCH_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
PIN=9328aff8a6c470b026aee7cefe53a0527de44814
NATIVE_CPU=${TENSORFOLD_NATIVE_CPU:-apple_m4}
if [[ ! -d "$SOURCE/.git" ]]; then
  git clone --no-checkout https://github.com/BobClawblaw/TensorFold.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$PIN"
fi
if [[ $(git -C "$SOURCE" rev-parse HEAD) != "$PIN" ]]; then
  echo "Refusing a source checkout that does not match pinned native port $PIN" >&2
  exit 1
fi
if ! grep -q 'extra.sum != one.sum' "$SOURCE/zig/src/core/row_projection.zig"; then
  echo 'Unrecognized mixed-quantization source' >&2
  exit 1
fi
if grep -q 'extra.group != one.group' "$SOURCE/zig/src/core/row_projection.zig"; then
  python3 "$PATCH_DIR/patch_mixed_shared.py" "$SOURCE"
fi
python3 "$PATCH_DIR/apply.py" "$SOURCE"
bash "$PATCH_DIR/test.sh" "$SOURCE" "$ZIG"
cd "$SOURCE"
"$ZIG" build native -Doptimize=ReleaseFast -Dcpu="$NATIVE_CPU" -j4
shasum -a 256 zig-out/native/bin/tensorfold-native
