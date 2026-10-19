#!/usr/bin/env bash
# Install IndexTTS-2.5 as a service on the AI server. Run server_check.sh first; this needs about 20 GB and a few
# minutes of downloading (11 GB of weights). Everything goes under one folder and nothing is installed system-wide
# except uv (into ~/.local/bin) if it is missing.
#
#   1. From your development machine, copy the four files into the folder you want (say ~/indextts-server on the server):
#        scp interpreter/local_tts/{server_check.sh,server_install.sh,indextts_server.py} \
#            interpreter/bakeoff/reference/neutral_ref.wav USER@SERVER:~/indextts-server/
#      (make the folder first:  ssh USER@SERVER mkdir -p ~/indextts-server)
#   2. On the server:   bash ~/indextts-server/server_install.sh
#   3. Start it:        bash ~/indextts-server/start.sh          (this script writes start.sh)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT=d9e41aac89fd00b3d71497fddb287b7f24613712      # the index-tts commit the pipeline was developed against
for f in indextts_server.py neutral_ref.wav; do
  [ -f "$HERE/$f" ] || { echo "missing $HERE/$f: copy it from the development machine (see the top of this script)"; exit 1; }
done

step() { printf '\n== %s\n' "$*"; }
export PATH="$HOME/.local/bin:$PATH"

step "uv"
if command -v uv >/dev/null 2>&1; then uv --version; else curl -LsSf https://astral.sh/uv/install.sh | sh; fi

step "IndexTTS source at $COMMIT"
if [ ! -d "$HERE/index-tts/.git" ]; then git clone https://github.com/index-tts/index-tts.git "$HERE/index-tts"; fi
git -C "$HERE/index-tts" fetch --quiet origin "$COMMIT" 2>/dev/null || git -C "$HERE/index-tts" fetch --quiet origin
git -C "$HERE/index-tts" checkout --quiet "$COMMIT"

step "Python environment (PyTorch 2.8 for CUDA 12.8; several minutes)"
( cd "$HERE/index-tts" && UV_PROJECT_ENVIRONMENT="$HERE/venv" uv sync )     # plain sync: the optional accel extras need CUDA_HOME

step "Weights (about 11 GB)"
HF_HOME="$HERE/hf" "$HERE/venv/bin/hf" download IndexTeam/IndexTTS-2.5 --local-dir "$HERE/models"

step "start.sh"
cat > "$HERE/start.sh" <<EOF
#!/usr/bin/env bash
# Starts the IndexTTS service on 127.0.0.1:\${INDEXTTS_PORT:-8790} (no authentication, so reach it through an SSH tunnel).
cd "$HERE/index-tts"
export HF_HOME="$HERE/hf" PYTHONPATH="$HERE/index-tts"
exec "$HERE/venv/bin/python" "$HERE/indextts_server.py" --repo "$HERE/index-tts" --model-dir "$HERE/models" \\
     --reference "$HERE/neutral_ref.wav" --host 127.0.0.1 --port "\${INDEXTTS_PORT:-8790}"
EOF
chmod +x "$HERE/start.sh"

step "done"
echo "Start the service:  bash $HERE/start.sh        (loading takes about a minute; it prints 'ready')"
echo "Keep it running with:  tmux new -s indextts 'bash $HERE/start.sh'    (or nohup ... &)"
echo "Then, from your development machine:"
echo "  ssh -N -L 8790:127.0.0.1:8790 USER@SERVER          # leave this running"
echo "  curl http://127.0.0.1:8790/health                   # should print {\"ok\": true, ...}"
