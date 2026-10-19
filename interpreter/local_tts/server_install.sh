#!/usr/bin/env bash
# Install IndexTTS-2.5 as a service on this machine (run it ON the AI server). Run server_check.sh first. It needs about
# 20 GB and some minutes of downloading (11 GB of weights). Everything goes into one folder (default ~/indextts-server);
# nothing is installed system-wide except uv (into ~/.local/bin) if it is missing.
#
#   git clone -b feature/sign-language-project https://github.com/json0130/sign-language-interpreter-.git ~/sli
#   bash ~/sli/interpreter/local_tts/server_check.sh
#   bash ~/sli/interpreter/local_tts/server_install.sh [install-folder]
#   bash ~/indextts-server/start.sh
set -euo pipefail
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # the repo's interpreter/local_tts folder
DEST="${1:-$HOME/indextts-server}"
COMMIT=d9e41aac89fd00b3d71497fddb287b7f24613712                # the index-tts commit the pipeline was developed against
for f in indextts_server.py assets/neutral_ref.wav; do
  [ -f "$SRC/$f" ] || { echo "missing $SRC/$f: run this from a clone of the sign-language-interpreter- repo"; exit 1; }
done

step() { printf '\n== %s\n' "$*"; }
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$DEST"
cp "$SRC/indextts_server.py" "$SRC/assets/neutral_ref.wav" "$DEST/"

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
