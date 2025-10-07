#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8000}"

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn[standard] python-multipart

mkdir -p uploads

exec uvicorn ML.server:app --host 0.0.0.0 --port "$PORT" --reload


