"""SMA's run-time fix for LiveCodeBench in NVIDIA's eval-factory container (livecodebench:26.03).

Its runner grades each test with --timeout, the HTTP request timeout, and never reads
--evaluation_timeout (LiveCodeBench's 6 s per test). SMA gives requests hours, so long answers can
finish, which also gave each test hours: an answer that never stops held a grading process for 4 hours,
and a correct answer slower than 6 s passed (LFM3-1.7B on SGLang, 2026-10-31).

SMA mounts this folder and puts it on PYTHONPATH, so Python imports this file in every process of the
container. It only acts when the runner's scenario_router module loads: grading then gets
--evaluation_timeout as its timeout. Requests keep --timeout. With SMA_GRADING_LOCK and SMA_GRADING_CPUS set
(grading_turn), grading also waits for the machine's grading lock and runs on the timed CPUs.
"""

import importlib.abc
import importlib.machinery
import sys

_TARGET = "livecodebench.runner.scenario_router"


def _patch(module):
    original = module.get_metrics

    def get_metrics(scenario, args, *rest, **kwargs):
        import copy

        graded = copy.copy(args)
        graded.timeout = args.evaluation_timeout
        print(f"SMA patch: grading with --evaluation_timeout {graded.timeout} s per test, not --timeout {args.timeout} s", file=sys.stderr, flush=True)
        from sma_grading import grading_turn  # Beside this file, on PYTHONPATH.

        with grading_turn():
            return original(scenario, graded, *rest, **kwargs)

    module.get_metrics = get_metrics


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def _run_hidden_sitecustomize():
    """Run the sitecustomize this file hides (the image's own, Ubuntu's apport hook), as Python would have."""
    import importlib.util
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if os.path.abspath(entry or ".") == here:
            continue
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [entry])
        if spec is not None and spec.loader is not None:
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            return


sys.meta_path.insert(0, _Finder())
_run_hidden_sitecustomize()
