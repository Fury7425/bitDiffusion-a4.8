from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class EntrypointSmokeTests(unittest.TestCase):
    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def test_package_entrypoints_show_help(self) -> None:
        for command in (
            ("-m", "bitdiffusion.sample", "--help"),
            ("-m", "bitdiffusion.export", "--help"),
        ):
            result = self._run(*command)
            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_root_wrappers_show_help(self) -> None:
        for command in (
            ("sample.py", "--help"),
            ("export.py", "--help"),
        ):
            result = self._run(*command)
            self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
