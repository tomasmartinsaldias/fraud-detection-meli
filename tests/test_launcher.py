"""Check environment reuse and conditional dependency installation without downloads."""

import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("project_launcher", ROOT / "scripts/run.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch.object(launcher.sys, "version_info", SimpleNamespace(major=3, minor=13)),
            patch.object(launcher.sys, "prefix", str(launcher.ENV)),
            patch.object(launcher.sys, "argv", ["run.py", "src/make_dataset.py", "--help"]),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def versions(self, name):
        return dict(line.split("==") for line in launcher.REQUIREMENTS.read_text().splitlines())[name]

    def test_ready_environment_does_not_install_or_recreate(self):
        with patch.object(launcher.importlib.metadata, "version", side_effect=self.versions), \
             patch.object(launcher.subprocess, "run") as run, \
             patch.object(launcher.subprocess, "call", return_value=7) as call, \
             patch.object(launcher.venv, "create") as create:
            self.assertEqual(launcher.main(), 7)
            create.assert_not_called()
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[0][-2:], ["pip", "check"])
            self.assertEqual(call.call_args.args[0][-1], "--help")

    def test_missing_or_wrong_version_installs_before_check(self):
        for installed in ["0.0", None]:
            with self.subTest(installed=installed), \
                 patch.object(launcher.importlib.metadata, "version") as version, \
                 patch.object(launcher.subprocess, "run") as run, \
                 patch.object(launcher.subprocess, "call", return_value=0):
                if installed is None:
                    version.side_effect = launcher.importlib.metadata.PackageNotFoundError
                else:
                    version.return_value = installed
                self.assertEqual(launcher.main(), 0)
                self.assertEqual(run.call_count, 2)
                self.assertIn("install", run.call_args_list[0].args[0])
                self.assertEqual(run.call_args_list[1].args[0][-2:], ["pip", "check"])

    def test_wrong_python_stops_before_running_script(self):
        with patch.object(launcher.sys, "version_info", SimpleNamespace(major=3, minor=14)), \
             patch.object(launcher.subprocess, "call") as call:
            self.assertEqual(launcher.main(), 1)
            call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
