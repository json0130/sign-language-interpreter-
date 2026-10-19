import subprocess
import tempfile
import unittest
from pathlib import Path

from interpreter.local_tts.make_paste import check_text, install_text

LOCAL = Path(__file__).resolve().parents[1] / "local_tts"


def run_bash(script: str, cwd: Path, home: Path) -> None:
    subprocess.run(["bash"], input=script, text=True, cwd=cwd, env={"HOME": str(home), "PATH": "/usr/bin:/bin"},
                   check=True, capture_output=True)


class TestServerPaste(unittest.TestCase):
    def test_check_paste_recreates_the_script(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            run_bash(check_text().replace("\nbash ~/server_check.sh\n", "\n"), home, home)   # write it, do not run it
            self.assertEqual((home / "server_check.sh").read_bytes(), (LOCAL / "server_check.sh").read_bytes())

    def test_install_paste_recreates_both_files(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            lines = install_text().splitlines(keepends=True)
            run_bash("".join(lines[:-1]), home, home)               # everything except the final "bash server_install.sh"
            for name in ("indextts_server.py", "server_install.sh"):
                self.assertEqual((home / "indextts-server" / name).read_bytes(), (LOCAL / name).read_bytes(), name)

    def test_pasted_text_has_no_stray_end_marker(self):
        for text in (check_text(), install_text()):
            self.assertEqual(text.count("END_OF_FILE_7Q'"), text.count("<<'END_OF_FILE_7Q'"))


if __name__ == "__main__":
    unittest.main()
