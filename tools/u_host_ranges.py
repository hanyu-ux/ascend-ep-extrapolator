#!/usr/bin/env python3
"""Add opt-in Host function ranges to an existing Model Runner V2 trace.

Run in the NPU sampling environment, using the original service command:
    python tools/u_host_ranges.py -- vllm serve ...

Real, nonempty execute_model calls enable U_HOST ranges around existing Host
functions. Capture, dummy and profiling calls suppress the ranges. The launcher
uses the installed torch record_function API and adds no profiler session.
"""

import argparse
import contextvars
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import textwrap


CONFIG_ENV = "U_HOST_RANGES_BOOTSTRAP"
MODULES = frozenset(("vllm.v1.worker.gpu.dp_utils", "vllm.v1.worker.gpu.model_runner",
                     "vllm_ascend.worker.v2.model_runner"))


def marker(event, **fields):
    try:
        print(event + " " + json.dumps(fields, ensure_ascii=True), file=sys.stderr, flush=True)
    except Exception:
        pass


class Controller:
    def __init__(self):
        self.active = contextvars.ContextVar("u_host_active", default=False)
        self.suppressed = contextvars.ContextVar("u_host_suppressed", default=False)
        self.hooks = {}
        self.errors = set()
        self.announced_pid = None
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self.after_fork)

    def after_fork(self):
        self.active.set(False)
        self.suppressed.set(False)
        self.errors = set()
        self.announced_pid = None

    def observation_error(self, stage, error, label):
        key = stage, label
        if key not in self.errors:
            self.errors.add(key)
            marker("U_HOST_OBSERVATION_ERROR", pid=os.getpid(), stage=stage,
                   hook=label, error=type(error).__name__)

    def scope_enter(self, label):
        if not self.active.get() or self.suppressed.get():
            return None
        try:
            torch = sys.modules.get("torch")
            factory = getattr(getattr(torch, "profiler", None), "record_function", None)
            if factory is None:
                self.observation_error("record_function", AttributeError("record_function unavailable"), label)
                return None
            scope = factory("U_HOST/" + label)
            scope.__enter__()
            return scope
        except Exception as error:
            self.observation_error("enter", error, label)
            return None

    def wrap(self, owner, name, mode="range"):
        original = getattr(owner, name, None)
        label = getattr(owner, "__name__", type(owner).__name__) + "." + name
        if label in self.hooks:
            return
        available = callable(original)
        self.hooks[label] = available
        marker("U_HOST_HOOK", pid=os.getpid(), hook=label, available=available)
        if not available or getattr(original, "__u_host_ranges__", False):
            return
        try:
            signature = inspect.signature(original) if mode == "execute" else None
        except Exception as error:
            self.hooks[label] = False
            self.observation_error("signature", error, label)
            return

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            active_token = suppress_token = scope = None
            if mode == "execute":
                enabled = False
                try:
                    bound = signature.bind_partial(*args, **kwargs)
                    bound.apply_defaults()
                    values = bound.arguments
                    total = getattr(values.get("scheduler_output"), "total_num_scheduled_tokens", None)
                    enabled = (type(total) is int and total > 0 and values.get("dummy_run") is not True
                               and values.get("is_profile") is not True and not self.suppressed.get())
                except Exception as error:
                    self.observation_error("bind", error, label)
                active_token = self.active.set(enabled)
                if enabled and self.announced_pid != os.getpid():
                    self.announced_pid = os.getpid()
                    marker("U_HOST_ACTIVE", pid=os.getpid(), hooks=self.hooks)
                scope = self.scope_enter(label)
            elif mode == "suppress":
                active_token = self.active.set(False)
                suppress_token = self.suppressed.set(True)
            else:
                scope = self.scope_enter(label)
            try:
                return original(*args, **kwargs)
            finally:
                if scope is not None:
                    try:
                        scope.__exit__(*sys.exc_info())
                    except Exception as error:
                        self.observation_error("exit", error, label)
                if suppress_token is not None:
                    self.suppressed.reset(suppress_token)
                if active_token is not None:
                    self.active.reset(active_token)

        wrapped.__u_host_ranges__ = True
        setattr(owner, name, wrapped)

    def module_loaded(self, module):
        name = module.__name__
        for function in ("async_copy_to_gpu", "build_attn_state", "expand_idx_mapping",
                         "dispatch_cg_and_sync_dp", "sync_cudagraph_and_dp_padding"):
            if hasattr(module, function) or name.endswith("dp_utils"):
                self.wrap(module, function)
        cls = getattr(module, "NPUModelRunner" if name.startswith("vllm_ascend") else "GPUModelRunner", None)
        if cls is None:
            return
        if name.startswith("vllm_ascend"):
            self.wrap(cls, "execute_model", mode="execute")
        for method in ("capture_model", "_dummy_run", "profile_run"):
            self.wrap(cls, method, mode="suppress")
        for method in ("update_states", "gather_batch_req_state", "prepare_inputs", "_update_seq_lens_cpu",
                       "_pad_query_start_loc_for_fia", "_pad_adaptive_query_start_loc_for_fia"):
            self.wrap(cls, method)


class ObserverLoader(importlib.abc.Loader):
    def __init__(self, original, controller):
        self.original, self.controller = original, controller

    def create_module(self, spec):
        create = getattr(self.original, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.original.exec_module(module)
        try:
            self.controller.module_loaded(module)
        except Exception as error:
            marker("U_HOST_OBSERVATION_ERROR", pid=os.getpid(), stage="install", error=type(error).__name__)

    def __getattr__(self, name):
        return getattr(self.original, name)


class ObserverFinder(importlib.abc.MetaPathFinder):
    def __init__(self, controller):
        self.controller = controller
        self._u_host_ranges = True
        self.finding = set()
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self.finding.clear)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in MODULES or fullname in self.finding:
            return None
        self.finding.add(fullname)
        try:
            for finder in tuple(sys.meta_path):
                method = getattr(finder, "find_spec", None) if finder is not self else None
                if method is None:
                    continue
                spec = method(fullname, path, target)
                if spec is not None:
                    if spec.loader is not None and hasattr(spec.loader, "exec_module"):
                        spec.loader = ObserverLoader(spec.loader, self.controller)
                    return spec
            return None
        finally:
            self.finding.discard(fullname)


def install():
    for finder in sys.meta_path:
        if getattr(finder, "_u_host_ranges", False):
            return finder.controller
    controller = Controller()
    sys.meta_path.insert(0, ObserverFinder(controller))
    for name in sorted(MODULES):
        if name in sys.modules:
            try:
                controller.module_loaded(sys.modules[name])
            except Exception as error:
                marker("U_HOST_OBSERVATION_ERROR", pid=os.getpid(), stage="install", error=type(error).__name__)
    marker("U_HOST_BOOTSTRAP", pid=os.getpid())
    return controller


def bootstrap_source(script):
    return textwrap.dedent(f"""\
        import importlib.machinery as _machinery
        import importlib.util as _util
        import os as _os
        import runpy as _runpy
        import sys as _sys
        _here = _os.path.dirname(__file__)
        _paths = [p for p in _sys.path if _os.path.realpath(p or '.') != _os.path.realpath(_here)]
        _previous = _machinery.PathFinder.find_spec('sitecustomize', _paths)
        if _previous is not None and _previous.loader is not None:
            _module = _util.module_from_spec(_previous)
            _sys.modules['sitecustomize'] = _module
            _previous.loader.exec_module(_module)
        try:
            _runpy.run_path({str(script)!r}, run_name='_u_host_bootstrap')
        except Exception as _error:
            print('U_HOST_OBSERVATION_ERROR bootstrap=' + type(_error).__name__, file=_sys.stderr, flush=True)
    """)


def launch(command):
    if CONFIG_ENV in os.environ:
        raise ValueError("Nested Host-range launch detected")
    with tempfile.TemporaryDirectory(prefix="u_host_ranges_") as temporary:
        bootstrap = Path(temporary)
        (bootstrap / "sitecustomize.py").write_text(bootstrap_source(Path(__file__).resolve()), encoding="utf-8")
        env = dict(os.environ)
        env[CONFIG_ENV] = "1"
        env["PYTHONPATH"] = str(bootstrap) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        own_session = os.name == "posix"
        process = subprocess.Popen(command, env=env, start_new_session=own_session)
        handlers = {}

        def forward(signum, _frame):
            if process.poll() is None:
                try:
                    if own_session:
                        os.killpg(process.pid, signum)
                    else:
                        process.send_signal(signum)
                except OSError:
                    pass

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.signal(signum, forward)
            code = process.wait()
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
        return code if code >= 0 else 128 - code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply the original service or sampling command after --")
    return launch(command)


if __name__ == "_u_host_bootstrap":
    install()
elif __name__ == "__main__":
    sys.exit(main())
