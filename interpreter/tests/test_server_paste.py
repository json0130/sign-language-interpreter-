import subprocess
import tempfile
import unittest
from pathlib import Path

from interpreter.local_tts.make_paste import check_text, install_text

LOCAL = Path(__file__).resolve().parents[1] / "local_tts"


def run_bash(script: str, cwd: Path, home: Path, **env) -> None:
    subprocess.run(["bash"], input=script, text=True, cwd=cwd, check=True, capture_output=True,
                   env={"HOME": str(home), "TMPDIR": str(home), "PATH": "/usr/bin:/bin", **env})


class TestServerPaste(unittest.TestCase):
    def test_check_paste_recreates_the_script(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            lines = check_text().splitlines(keepends=True)
            run_bash("".join(lines[:-1]), home, home)                  # write it, do not run it
            self.assertEqual((home / "server_check.sh").read_bytes(), (LOCAL / "server_check.sh").read_bytes())

    def test_install_paste_recreates_both_files(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)
            lines = install_text().splitlines(keepends=True)
            run_bash("".join(lines[:-1]), home, home)               # everything except the final "bash server_install.sh"
            for name in ("indextts_server.py", "server_install.sh"):
                self.assertEqual((home / "indextts-server" / name).read_bytes(), (LOCAL / name).read_bytes(), name)

    def test_install_folder_can_be_chosen(self):
        with tempfile.TemporaryDirectory() as d:
            home, big = Path(d), Path(d) / "big" / "volume"
            lines = install_text().splitlines(keepends=True)
            run_bash("".join(lines[:-1]), home, home, INDEXTTS_DIR=str(big / "indextts-server"))
            self.assertTrue((big / "indextts-server" / "server_install.sh").exists())
            self.assertFalse((home / "indextts-server").exists())

    def test_installer_keeps_everything_out_of_home(self):
        text = (LOCAL / "server_install.sh").read_text()
        for needle in ('UV_CACHE_DIR="$DEST/uv-cache"', 'UV_PYTHON_INSTALL_DIR="$DEST/python"', 'UV_UNMANAGED_INSTALL="$DEST/bin"',
                       'HF_HOME="$DEST/hf"'):
            self.assertIn(needle, text)
        self.assertNotIn(".local/bin", text)

    def test_check_creates_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "not" / "yet" / "there"
            before = sorted(p.name for p in Path(d).rglob("*"))
            subprocess.run(["bash", str(LOCAL / "server_check.sh"), str(target)], capture_output=True, text=True,
                           env={"HOME": d, "PATH": "/usr/bin:/bin", "SKIP_NETWORK": "1"})
            self.assertFalse(target.exists())
            self.assertEqual(sorted(p.name for p in Path(d).rglob("*")), before)

    def test_pasted_text_has_no_stray_end_marker(self):
        for text in (check_text(), install_text()):
            self.assertEqual(text.count("END_OF_FILE_7Q'"), text.count("<<'END_OF_FILE_7Q'"))


if __name__ == "__main__":
    unittest.main()
