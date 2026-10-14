#!/usr/bin/env bash
# Commands that worked for the Step B2 bake-off, in the order they were run (kept for reproducibility; read before
# running). One venv per candidate under venvs/ so dependencies cannot conflict. Needs uv, git, espeak-ng, ffmpeg.
# GPU: RTX 5060 (Blackwell, sm_120), so every venv needs torch >= 2.7 with CUDA 12.8+ kernels.
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME=$PWD/models/hf
mkdir -p venvs models out repos

# Kokoro (neutral baseline). The spaCy model that misaki wants is not auto-installed into a uv venv.
uv venv --python 3.11 venvs/kokoro
uv pip install --python venvs/kokoro/bin/python "kokoro>=0.9.2" soundfile \
  "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"

# Parler-TTS (installs transformers 4.46.1 itself; torch overridden for Blackwell)
uv venv --python 3.11 venvs/parler
uv pip install --python venvs/parler/bin/python --override overrides_torch.txt \
  "git+https://github.com/huggingface/parler-tts.git" soundfile

# Chatterbox: pins torch==2.6.0 (no sm_120 kernels), so overridden; perth's watermarker needs pkg_resources.
uv venv --python 3.11 venvs/chatterbox
uv pip install --python venvs/chatterbox/bin/python --override overrides_torch.txt chatterbox-tts soundfile "setuptools<81"

# CosyVoice 3: repo requirements pin torch 2.3.1 etc., so a trimmed inference-only list is used instead.
git clone --depth 1 --recursive https://github.com/FunAudioLLM/CosyVoice.git repos/CosyVoice
uv venv --python 3.10 venvs/cosyvoice
uv pip install --python venvs/cosyvoice/bin/python -r cosyvoice_requirements.txt
venvs/cosyvoice/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='models/cosyvoice3')"

# IndexTTS-2.5: `uv sync --all-extras` (README) fails building flash-attn/deepspeed (no CUDA_HOME); plain `uv sync` works.
git clone --depth 1 https://github.com/index-tts/index-tts.git repos/index-tts
(cd repos/index-tts && UV_PROJECT_ENVIRONMENT=../../venvs/indextts uv sync)
venvs/indextts/bin/hf download IndexTeam/IndexTTS-2.5 --local-dir models/indextts-2.5

# Speech-emotion recogniser (rough proxy) and Whisper (Mandarin round-trip check; whisper comes from the cosyvoice venv)
uv venv --python 3.11 venvs/ser
uv pip install --python venvs/ser/bin/python funasr modelscope huggingface_hub torch torchaudio soundfile numpy

# Render everything, normalise loudness, run the SER, print the table (python3 = stdlib only):
#   python3 run.py --device cuda
# Mandarin smoke check (one venv per engine), then the Whisper transcription:
#   for e in chatterbox cosyvoice indextts; do PYTHONPATH=../.. venvs/$e/bin/python zh_check.py $e; done
#   venvs/cosyvoice/bin/python zh_check.py asr
