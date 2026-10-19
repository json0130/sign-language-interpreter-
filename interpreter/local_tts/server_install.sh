#!/usr/bin/env bash
# Install IndexTTS-2.5 as a service on this machine (run it ON the AI server). Run server_check.sh first. It needs about
# 20 GB and some minutes of downloading (11 GB of weights). Everything goes into one folder (default ~/indextts-server);
# nothing is installed system-wide except uv (into ~/.local/bin) if it is missing.
#
# It needs two project files next to it (or in ./assets, or in the current folder): indextts_server.py and neutral_ref.wav.
# From a clone of the repo they are found automatically; without the repo, put them in the same folder as this script.
# If neutral_ref.wav is missing it is downloaded from the public repo; if that fails you are told what to copy.
#
#   bash server_install.sh [install-folder]        then        bash ~/indextts-server/start.sh
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
DEST="${1:-$HOME/indextts-server}"
COMMIT=d9e41aac89fd00b3d71497fddb287b7f24613712                # the index-tts commit the pipeline was developed against
RAW=https://raw.githubusercontent.com/json0130/sign-language-interpreter-/feature/sign-language-project/interpreter/local_tts
find_file() { for d in "$SRC" "$SRC/assets" "$PWD"; do [ -f "$d/$1" ] && { echo "$d/$1"; return 0; }; done; return 1; }
mkdir -p "$DEST"
SERVER_PY="$(find_file indextts_server.py)" || { echo "missing indextts_server.py: put it next to this script (it is in the repo at interpreter/local_tts/)"; exit 1; }
if ! REF="$(find_file neutral_ref.wav)"; then
  echo "neutral_ref.wav not found next to the script; trying to download it"
  curl -fsSL -o "$DEST/neutral_ref.wav.part" "$RAW/assets/neutral_ref.wav" && mv "$DEST/neutral_ref.wav.part" "$DEST/neutral_ref.wav" \
    || { rm -f "$DEST/neutral_ref.wav.part"; echo "could not download it. Copy this 160 KB file from the development machine to $DEST/neutral_ref.wav (interpreter/local_tts/assets/neutral_ref.wav) and run this script again."; exit 1; }
  REF="$DEST/neutral_ref.wav"
fi

step() { printf '\n== %s\n' "$*"; }
export PATH="$HOME/.local/bin:$PATH"
[ "$SERVER_PY" -ef "$DEST/indextts_server.py" ] || cp "$SERVER_PY" "$DEST/indextts_server.py"
[ "$REF" -ef "$DEST/neutral_ref.wav" ] || cp "$REF" "$DEST/neutral_ref.wav"

step "uv"
if command -v uv >/dev/null 2>&1; then uv --version; else curl -LsSf https://astral.sh/uv/install.sh | sh; fi

step "IndexTTS source at $COMMIT"
if [ ! -d "$DEST/index-tts/.git" ]; then git clone https://github.com/index-tts/index-tts.git "$DEST/index-tts"; fi
git -C "$DEST/index-tts" fetch --quiet origin "$COMMIT" 2>/dev/null || git -C "$DEST/index-tts" fetch --quiet origin
git -C "$DEST/index-tts" checkout --quiet "$COMMIT"

step "Python environment (PyTorch 2.8 for CUDA 12.8; several minutes)"
( cd "$DEST/index-tts" && UV_PROJECT_ENVIRONMENT="$DEST/venv" uv sync )     # plain sync: the optional accel extras need CUDA_HOME

step "Weights (about 11 GB)"
HF_HOME="$DEST/hf" "$DEST/venv/bin/hf" download IndexTeam/IndexTTS-2.5 --local-dir "$DEST/models"

step "start.sh"
cat > "$DEST/start.sh" <<EOF
#!/usr/bin/env bash
# Starts the IndexTTS service on 127.0.0.1:\${INDEXTTS_PORT:-8790} (no authentication, so reach it through an SSH tunnel).
cd "$DEST/index-tts"
export HF_HOME="$DEST/hf" PYTHONPATH="$DEST/index-tts"
exec "$DEST/venv/bin/python" "$DEST/indextts_server.py" --repo "$DEST/index-tts" --model-dir "$DEST/models" \\
     --reference "$DEST/neutral_ref.wav" --host 127.0.0.1 --port "\${INDEXTTS_PORT:-8790}"
EOF
chmod +x "$DEST/start.sh"

step "done"
echo "Start the service:  bash $DEST/start.sh        (loading takes about a minute; it prints 'ready')"
echo "Keep it running:    tmux new -s indextts 'bash $DEST/start.sh'"
echo "Test on this machine:  curl http://127.0.0.1:8790/health"
echo "From your development machine:  ssh -N -L 8790:127.0.0.1:8790 USER@SERVER   then   curl http://127.0.0.1:8790/health"
