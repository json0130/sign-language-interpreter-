#!/usr/bin/env bash
# Can IndexTTS-2.5 be installed and run on this machine? READ-ONLY: it installs nothing and changes nothing.
#
#   On the AI server:        bash server_check.sh [install-dir]      (install-dir: where it would go; default $INDEXTTS_DIR or $HOME)
#   From your own machine:   ssh USER@SERVER 'bash -s' < interpreter/local_tts/server_check.sh
#   In a sandbox that cannot reach the GPU driver or the network, set SKIP_NETWORK=1 to skip the download checks.
# It creates no files or folders (the disk check measures the nearest folder that already exists).
#
# Prints PASS / WARN / FAIL per check and a verdict. Exit status 0 = no FAIL. The numbers come from the install done on
# the development machine (RTX 5060, 8 GB): peak GPU memory 5.7 GB while speaking, 8.3 GB environment, 11 GB weights.
# Compute capability and driver limits for PyTorch 2.8 with CUDA 12.8 are from PyTorch's documentation, not tested here.

DIR="${1:-${INDEXTTS_DIR:-$HOME}}"
PORT="${INDEXTTS_PORT:-8790}"
NEED_FREE_MIB=6500        # free GPU memory to run comfortably (peak use was 5.7 GB = about 5800 MiB)
MIN_FREE_MIB=6000
NEED_DISK_GB=25           # 8.3 GB environment + 11 GB weights + headroom
NEED_RAM_GB=8
fails=0; warns=0

pass() { printf '  PASS  %s\n' "$*"; }
warn() { printf '  WARN  %s\n' "$*"; warns=$((warns + 1)); }
fail() { printf '  FAIL  %s\n' "$*"; fails=$((fails + 1)); }
have() { command -v "$1" >/dev/null 2>&1; }

echo "IndexTTS server preflight on $(hostname), $(date '+%Y-%m-%d %H:%M')  (install dir: $DIR)"
echo "OS: $(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -sr), kernel $(uname -r), $(uname -m)"

echo; echo "GPU"
if ! have nvidia-smi; then
  fail "nvidia-smi not found: no NVIDIA driver, so no CUDA GPU"
else
  gpu=$(nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version,compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -z "$gpu" ]; then
    fail "nvidia-smi found no GPU"
  else
    IFS=',' read -r name total used driver cap <<<"$gpu"
    name=$(echo "$name" | xargs); total=$(echo "$total" | xargs); used=$(echo "$used" | xargs)
    driver=$(echo "$driver" | xargs); cap=$(echo "$cap" | xargs)
    free=$((total - used))
    echo "  GPU: $name, ${total} MiB total, ${used} MiB used, ${free} MiB free; driver $driver; compute capability $cap"
    if   [ "$free" -ge "$NEED_FREE_MIB" ]; then pass "free GPU memory ${free} MiB (needs about 5800, ${NEED_FREE_MIB}+ is comfortable)"
    elif [ "$free" -ge "$MIN_FREE_MIB" ];  then warn "free GPU memory ${free} MiB is tight (needs about 5800); close other GPU programs"
    elif [ "$total" -ge "$NEED_FREE_MIB" ]; then warn "only ${free} MiB free of ${total} MiB: something else is using the GPU (listed below); it would fit if that were closed"
    else fail "GPU memory ${total} MiB total is below the ${MIN_FREE_MIB} MiB needed"; fi
    drv_major=${driver%%.*}
    if [ "${drv_major:-0}" -ge 570 ] 2>/dev/null; then pass "driver $driver supports CUDA 12.8 (needs 570 or newer)"
    else fail "driver $driver is older than 570, which the CUDA 12.8 PyTorch build needs; update the NVIDIA driver"; fi
    cap_major=${cap%%.*}
    if [ "${cap_major:-0}" -ge 7 ] 2>/dev/null; then pass "compute capability $cap (7.0 or newer)"
    else warn "compute capability $cap is older than 7.0; PyTorch 2.8 with CUDA 12.8 may not support it (check before installing)"; fi
  fi
  users=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
  if [ -n "$users" ]; then
    echo "  Programs using the GPU now:"
    while IFS=',' read -r pid mib; do
      pid=$(echo "$pid" | xargs); mib=$(echo "$mib" | xargs)
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | awk '{print $1}' | awk -F/ '{print $(NF-1)"/"$NF}')
      printf '    pid %s %s: %s MiB\n' "$pid" "${cmd:-(unreadable)}" "$mib"
    done <<<"$users"
  fi
fi

echo; echo "Disk, memory, port"
probe="$DIR"
while [ ! -d "$probe" ] && [ "$probe" != "/" ]; do probe="$(dirname "$probe")"; done      # nearest folder that exists
free_gb=$(df -Pk "$probe" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}')
where="$DIR"; [ "$probe" != "$DIR" ] && where="$DIR (measured at $probe, which exists)"
if [ -z "$free_gb" ]; then fail "cannot read free space for $DIR"
elif [ "$free_gb" -ge "$NEED_DISK_GB" ]; then pass "disk: ${free_gb} GB free at $where (needs about 20, ${NEED_DISK_GB}+ is comfortable)"
else fail "disk: only ${free_gb} GB free at $where; needs about ${NEED_DISK_GB} GB. Try the big volume: bash server_check.sh /path/to/big/volume/indextts-server"; fi
ram_gb=$(awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
if [ -z "$ram_gb" ]; then warn "cannot read available RAM"
elif [ "$ram_gb" -ge "$NEED_RAM_GB" ]; then pass "RAM: ${ram_gb} GB available"
else warn "RAM: ${ram_gb} GB available (loading the model wants about ${NEED_RAM_GB} GB)"; fi
if have ss && ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${PORT}$"; then
  warn "port $PORT is already in use here (use another with INDEXTTS_PORT=NNNN, and set the same port in the pipeline's URL)"
else pass "port $PORT is free"; fi

echo; echo "Tools"
if have python3; then pass "python3 $(python3 --version 2>&1 | awk '{print $2}') (the install brings its own Python 3.11 via uv)"; else warn "python3 not found (only needed for uv's installer fallback)"; fi
if have uv; then pass "uv $(uv --version 2>&1 | awk '{print $2}')"; else warn "uv not installed: the install script will fetch it from astral.sh (needs that host reachable)"; fi
for t in git curl; do if have "$t"; then pass "$t"; else fail "$t not found (apt install $t)"; fi; done
if have ffmpeg; then pass "ffmpeg"; else warn "ffmpeg not found (apt install ffmpeg); the development machine has it"; fi
if have ssh; then pass "ssh (for the tunnel from your machine)"; fi

echo; echo "Network (the install downloads from these)"
for url in https://github.com https://huggingface.co https://pypi.org https://download.pytorch.org https://files.pythonhosted.org https://astral.sh; do
  if [ -n "${SKIP_NETWORK:-}" ]; then
    echo "  SKIP  $url (SKIP_NETWORK is set: not checked)"
  elif have curl; then
    code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' -I "$url" 2>/dev/null)
    case "$code" in
      2*|3*|4*) pass "$url reachable (HTTP $code)" ;;
      *) fail "$url not reachable (HTTP ${code:-none}): proxy or firewall? the install needs it" ;;
    esac
  fi
done

echo; echo "Verdict"
if [ "$fails" -gt 0 ]; then
  echo "  NOT READY: $fails FAIL, $warns WARN. Fix the FAIL lines above, then run this again."
  exit 1
elif [ "$warns" -gt 0 ]; then
  echo "  POSSIBLE: 0 FAIL, $warns WARN. Read the WARN lines; none blocks the install itself."
else
  echo "  READY: all checks passed."
fi
here="${BASH_SOURCE[0]:-}"
if [ -n "$here" ] && [ -f "$(dirname "$here")/server_install.sh" ]; then
  echo "  Next:  bash $(cd "$(dirname "$here")" && pwd)/server_install.sh $DIR"
else
  echo "  Next:  run the install step with the same folder:  export INDEXTTS_DIR=$DIR"
fi
exit 0
