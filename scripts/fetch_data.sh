#!/usr/bin/env bash
# Download XAUUSD M1 history (2016-09..2026-09, parquet) from Google Drive into data/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -fSL -o data/XAUUSD_41.parquet \
  "https://drive.usercontent.google.com/download?id=1S3AjTWTWzq76bUyHxlgTqNhZzVsOtgwf&export=download&confirm=t"
