"""Play a WAV with whichever system player exists."""
import shutil
import subprocess

# (executable, args before the path, args after)
PLAYERS = (
    ("afplay", [], []),
    ("paplay", [], []),
    ("aplay", ["-q"], []),
    ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"], []),
)


def play(path) -> bool:
    """Play `path` and wait. Returns False (after printing why) if no player is available."""
    for exe, pre, post in PLAYERS:
        if shutil.which(exe):
            subprocess.run([exe, *pre, str(path), *post], check=False)
            return True
    ps = shutil.which("powershell")
    if ps:
        subprocess.run([ps, "-c", f"(New-Object Media.SoundPlayer '{path}').PlaySync()"], check=False)
        return True
    print(f"No audio player found (tried afplay, paplay, aplay, ffplay, powershell). Open {path} yourself.")
    return False
