#!/bin/sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRADER="$SCRIPT_DIR/new.py"
DATASET="$SCRIPT_DIR/round4"
OUTPUT_DIR="$SCRIPT_DIR/backtests"

export PATH="$REPO_ROOT/prosperity_rust_backtester/target/release:$HOME/.cargo/bin:$PATH"

mkdir -p "$OUTPUT_DIR"

rust_backtester \
    --trader "$TRADER" \
    --dataset "$DATASET" \
    --output-root "$OUTPUT_DIR" \
    --persist \
    --artifact-mode full \
    --carry
