#!/usr/bin/env bash
# Humaneness Voice: local director and streaming speech application.
# Keep this thin: all model/adapter settings live in config.py and the manifest.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
for env_file in .env.generated .env.local; do
  if [[ -f "$env_file" ]]; then
    # Both files are local and gitignored. Never put secrets in .env.generated.
    set -a
    source "$env_file"
    set +a
  fi
done

DATA_DIR="${HUMANENESS_DATA_DIR:-${MOSS_DATA_DIR:-$HOME/.cache/humaneness-voice}}"
PYTHON_BIN="${MOSS_PYTHON:-$ROOT_DIR/.venv/bin/python}"
LLAMA_BIN="${LLAMA_SERVER:-}"
MOSS_LLM_GPU="${MOSS_LLM_GPU:-0}"
MOSS_TTS_GPU="${MOSS_TTS_GPU:-1}"
MOSS_APP_PORT="${MOSS_APP_PORT:-8792}"
MOSS_LLM_PORT="${MOSS_LLM_PORT:-8790}"
export MOSS_APP_PORT MOSS_LLM_GPU MOSS_TTS_GPU
export MOSS_LLM_BASE="${MOSS_LLM_BASE:-http://127.0.0.1:$MOSS_LLM_PORT}"
LOG_DIR="$ROOT_DIR/logs"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment missing: $PYTHON_BIN" >&2
  echo "See docs/INSTALL.md: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

find_director_file() {
  local pattern="$1"
  local matches=()
  shopt -s nullglob
  matches=("$DATA_DIR"/director/$pattern)
  shopt -u nullglob
  [[ ${#matches[@]} -gt 0 ]] && printf '%s\n' "${matches[0]}"
}

start_llm() {
  if [[ -z "$LLAMA_BIN" ]]; then
    LLAMA_BIN="$(command -v llama-server || true)"
  fi
  if [[ -z "$LLAMA_BIN" || ! -x "$LLAMA_BIN" ]]; then
    echo "llama-server is missing; set LLAMA_SERVER to your llama.cpp binary." >&2
    return 1
  fi
  local gguf="${MOSS_LLM_GGUF:-}"
  local mmproj="${MOSS_LLM_MMPROJ:-}"
  [[ -n "$gguf" ]] || gguf="$(find_director_file '*Q4_K_XL.gguf' || true)"
  [[ -n "$mmproj" ]] || mmproj="$(find_director_file 'mmproj-F16.gguf' || true)"
  if [[ -z "$gguf" || ! -f "$gguf" ]]; then
    echo "Director GGUF missing. Run: .venv/bin/python setup/fetch_all.py --director" >&2
    return 1
  fi
  local args=(--model "$gguf" --alias gemma-4-12b-it-qat
              --host 127.0.0.1 --port "$MOSS_LLM_PORT"
              --n-gpu-layers 999 --ctx-size "${MOSS_LLM_CTX:-16384}"
              --batch-size 512 --parallel 1 --flash-attn on --no-warmup
              --reasoning off --reasoning-budget 0)
  [[ -z "$mmproj" || ! -f "$mmproj" ]] || args+=(--mmproj "$mmproj")
  echo "[director] GPU $MOSS_LLM_GPU, port $MOSS_LLM_PORT"
  CUDA_VISIBLE_DEVICES="$MOSS_LLM_GPU" "$LLAMA_BIN" "${args[@]}" "$@"
}

start_app() {
  echo "[voice] GPU $MOSS_TTS_GPU, http://127.0.0.1:$MOSS_APP_PORT"
  env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES="$MOSS_TTS_GPU,$MOSS_LLM_GPU" \
    "$PYTHON_BIN" -m uvicorn app:app --host "${MOSS_APP_HOST:-127.0.0.1}" \
    --port "$MOSS_APP_PORT"
}

start_sidon() {
  CUDA_VISIBLE_DEVICES="$MOSS_LLM_GPU" "$PYTHON_BIN" -m uvicorn sidon_server:app \
    --host 127.0.0.1 --port "${MOSS_SIDON_PORT:-8793}"
}

stop_local() {
  for name in llm app sidon; do
    local pidfile="$LOG_DIR/$name.pid"
    if [[ -f "$pidfile" ]]; then
      local pid
      pid="$(<"$pidfile")"
      [[ "$pid" =~ ^[0-9]+$ ]] && kill "$pid" 2>/dev/null || true
      unlink "$pidfile"
    fi
  done
}

case "${1:-both}" in
  llm) start_llm ;;
  app) start_app ;;
  sidon) start_sidon ;;
  both)
    mkdir -p "$LOG_DIR"
    start_llm >"$LOG_DIR/llm.log" 2>&1 &
    echo $! >"$LOG_DIR/llm.pid"
    start_app >"$LOG_DIR/app.log" 2>&1 &
    echo $! >"$LOG_DIR/app.pid"
    echo "Humaneness Voice starting at http://127.0.0.1:$MOSS_APP_PORT"
    echo "Logs: $LOG_DIR/llm.log and $LOG_DIR/app.log"
    trap stop_local EXIT INT TERM
    wait
    ;;
  stop) stop_local ;;
  *) echo "usage: ./run.sh {both|llm|app|sidon|stop}" >&2; exit 2 ;;
esac
