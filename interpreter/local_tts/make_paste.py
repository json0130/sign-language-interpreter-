"""Write the AI-server setup as two plain-text files you can paste into a terminal on a machine that cannot reach the repo:

    python3 -m interpreter.local_tts.make_paste [out-dir]        (default: interpreter/audio_out/server_paste/)

    1_check.txt    pastes server_check.sh into ~/server_check.sh and runs it (read-only; installs nothing)
    2_install.txt  pastes indextts_server.py and server_install.sh into ~/indextts-server and runs the installer

The files are generated from the real scripts, so they cannot drift. Only the 160 KB voice clip is not text: the installer
downloads it, and says exactly what to copy if the download is blocked.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
END = "END_OF_FILE_7Q"     # not "EOF": server_install.sh has its own heredoc ending in EOF


def heredoc(dest: str, name: str) -> str:
    body = (HERE / name).read_text()
    if END in body:
        raise ValueError(f"{name} contains the marker {END}")
    return f"cat > {dest} <<'{END}'\n{body}{END}\n"


def check_text() -> str:
    return heredoc("~/server_check.sh", "server_check.sh") + "bash ~/server_check.sh\n"


def install_text() -> str:
    return ("mkdir -p ~/indextts-server && cd ~/indextts-server\n" + heredoc("indextts_server.py", "indextts_server.py")
            + heredoc("server_install.sh", "server_install.sh") + "bash server_install.sh\n")


def main(argv=None) -> None:
    out = Path((argv or sys.argv[1:] or [HERE.parent / "audio_out" / "server_paste"])[0])
    out.mkdir(parents=True, exist_ok=True)
    (out / "1_check.txt").write_text(check_text())
    (out / "2_install.txt").write_text(install_text())
    print(f"wrote {out}/1_check.txt and {out}/2_install.txt")


if __name__ == "__main__":
    main()
