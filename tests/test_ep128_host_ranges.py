"""Opt-in Host ranges preserve business execution and skip auxiliary work."""

import contextlib
import importlib.util
from importlib.machinery import ModuleSpec
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "u_host_ranges.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("u_host_ranges_test", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HostRangeTests(unittest.TestCase):
    def setUp(self):
        self.tool = load_tool()
        self.events = []
        self.failure = None
        self.scope_failure = None

        test = self

        class Scope:
            def __init__(self, name):
                self.name = name

            def __enter__(self):
                test.events.append(("enter", self.name))
                if test.scope_failure == "enter":
                    raise RuntimeError("diagnostic enter")

            def __exit__(self, *_args):
                test.events.append(("exit", self.name))
                if test.scope_failure == "exit":
                    raise RuntimeError("diagnostic exit")

        def record_function(name):
            if self.scope_failure == "factory":
                raise RuntimeError("diagnostic factory")
            return Scope(name)

        self.torch_patch = patch.dict(sys.modules, {"torch": SimpleNamespace(
            profiler=SimpleNamespace(record_function=record_function))})
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

        class Runner:
            calls = 0

            def prepare_inputs(self):
                self.calls += 1
                if test.failure is not None:
                    raise test.failure
                return "prepared"

            def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
                return self.prepare_inputs()

            def capture_model(self):
                return self.prepare_inputs()

            def _dummy_run(self):
                return self.prepare_inputs()

            def profile_run(self):
                return self.execute_model(test.scheduled())

        self.runner = Runner()
        self.controller = self.tool.Controller()
        with contextlib.redirect_stderr(io.StringIO()):
            self.controller.wrap(Runner, "prepare_inputs")
            self.controller.wrap(Runner, "execute_model", mode="execute")
            for name in ("capture_model", "_dummy_run", "profile_run"):
                self.controller.wrap(Runner, name, mode="suppress")

    def scheduled(self, tokens=8):
        return SimpleNamespace(total_num_scheduled_tokens=tokens)

    def test_only_real_nonempty_execute_records_host_ranges(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.runner.execute_model(self.scheduled()), "prepared")
        self.assertEqual([kind for kind, _ in self.events], ["enter", "enter", "exit", "exit"])
        self.assertTrue(all(name.startswith("U_HOST/") for _, name in self.events))
        self.assertIn("execute_model", self.events[0][1])
        self.assertIn("prepare_inputs", self.events[1][1])
        self.assertFalse(self.controller.active.get())
        self.events.clear()
        self.runner.execute_model(self.scheduled(0))
        self.runner.execute_model(self.scheduled(), dummy_run=True)
        self.runner.execute_model(self.scheduled(), is_profile=True)
        self.runner.execute_model(self.scheduled(), True, False)
        self.runner.prepare_inputs()
        self.assertEqual(self.events, [])
        self.assertEqual(self.runner.calls, 6)

    def test_capture_dummy_and_profile_suppress_inherited_active_context(self):
        token = self.controller.active.set(True)
        try:
            self.runner.prepare_inputs()
            for name in ("capture_model", "_dummy_run", "profile_run"):
                self.assertEqual(getattr(self.runner, name)(), "prepared")
                self.assertTrue(self.controller.active.get())
            self.runner.prepare_inputs()
        finally:
            self.controller.active.reset(token)
        self.assertEqual([kind for kind, _ in self.events], ["enter", "exit", "enter", "exit"])
        self.assertFalse(self.controller.active.get())

    def test_profiler_failures_preserve_return_value_and_business_exception(self):
        for mode in ("factory", "enter", "exit"):
            with self.subTest(mode=mode), contextlib.redirect_stderr(io.StringIO()):
                self.scope_failure = mode
                self.failure = None
                self.assertEqual(self.runner.execute_model(self.scheduled()), "prepared")
                expected = ValueError("business failure")
                self.failure = expected
                with self.assertRaises(ValueError) as caught:
                    self.runner.execute_model(self.scheduled())
                self.assertIs(caught.exception, expected)
                self.assertFalse(self.controller.active.get())

    def test_module_hook_wraps_normal_runner_entry_points_without_eager_imports(self):
        class NPUModelRunner:
            def execute_model(self, scheduler_output, dummy_run=False, is_profile=False):
                return self.prepare_inputs()

            def prepare_inputs(self):
                return 42

        module = SimpleNamespace(__name__="vllm_ascend.worker.v2.model_runner", NPUModelRunner=NPUModelRunner)
        with contextlib.redirect_stderr(io.StringIO()):
            self.controller.module_loaded(module)
            self.assertEqual(NPUModelRunner().execute_model(self.scheduled()), 42)
        self.assertEqual([kind for kind, _ in self.events], ["enter", "enter", "exit", "exit"])
        self.assertFalse(self.controller.active.get())

    def test_finder_coexists_with_a_delegating_observer_and_keeps_import_errors(self):
        name = "vllm.v1.worker.gpu.dp_utils"
        observer = self.tool.ObserverFinder(self.controller)
        executed = []
        expected = ValueError("business module execution")

        class Loader:
            fail = False

            def exec_module(self, module):
                if self.fail:
                    raise expected
                executed.append(module.__name__)

        loader = Loader()

        class ExistingObserver:
            def find_spec(self, fullname, path=None, target=None):
                for finder in tuple(sys.meta_path):
                    if finder is self:
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None:
                        return spec
                return None

        class TerminalFinder:
            def find_spec(self, fullname, path=None, target=None):
                return ModuleSpec(fullname, loader) if fullname == name else None

        existing, terminal = ExistingObserver(), TerminalFinder()
        for leader in (observer, existing):
            with self.subTest(leader=type(leader).__name__):
                order = [observer, existing, terminal] if leader is observer else [existing, observer, terminal]
                with patch.object(sys, "meta_path", order), contextlib.redirect_stderr(io.StringIO()):
                    spec = leader.find_spec(name)
                    self.assertIsNotNone(spec)
                    self.assertEqual(observer.finding, set())
                    spec.loader.exec_module(ModuleType(name))
                    loader.fail = True
                    with self.assertRaises(ValueError) as caught:
                        spec.loader.exec_module(ModuleType(name))
                    self.assertIs(caught.exception, expected)
                    loader.fail = False
        self.assertEqual(executed, [name, name])

    def test_launcher_chains_existing_sitecustomize_and_preserves_child_exit(self):
        with tempfile.TemporaryDirectory(prefix="ep128 Host ranges ") as temporary:
            base = Path(temporary)
            prior = base / "previous custom"
            prior.mkdir()
            (prior / "sitecustomize.py").write_text(
                "import os\nos.environ['U_HOST_PREVIOUS_CUSTOM_LOADED'] = 'yes'\n", encoding="utf-8")
            command = [sys.executable, "-B", "-S", str(TOOL), "--", sys.executable, "-B", "-c",
                       "import os,sys; assert os.environ.get('U_HOST_PREVIOUS_CUSTOM_LOADED') == 'yes'; "
                       "assert 'torch' not in sys.modules; print('child-ok'); sys.exit(7)"]
            env = {**os.environ, "PYTHONPATH": str(prior), "PYTHONIOENCODING": "utf-8"}
            result = subprocess.run(command, cwd=base, capture_output=True, text=True, encoding="utf-8",
                                    timeout=30, env=env)
            self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
            self.assertIn("child-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
