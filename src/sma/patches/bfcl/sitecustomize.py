"""SMA's run-time patch for BFCL in NVIDIA's eval-factory container (bfcl:26.03), for a suite with a limit.

BFCL's own --limit keeps the first questions after sorting every category by name, so it drops whole
categories (multi-turn among them); its help says "for testing only". When SMA sets SMA_QUESTION_LIMIT,
this patch makes `bfcl generate` answer an even share of every category instead (sma_subset.py), and makes
`bfcl evaluate` compare each answer with its own question: the evaluator pairs answers with questions and
ground truths by position, which only holds for a whole category. Without SMA_QUESTION_LIMIT it does nothing.

SMA mounts this folder and puts it on PYTHONPATH, so Python imports this file in every process of the container.
"""

import importlib.abc
import importlib.machinery
import os
import sys

_LIMIT = int(os.environ.get("SMA_QUESTION_LIMIT") or 0)


def _patch_generate(module):
    from sma_subset import even_share

    original = module.collect_test_cases

    def collect_test_cases(args, model_name, all_test_categories, all_test_file_paths, all_test_entries_involved):
        keep = even_share([entry["id"] for entry in all_test_entries_involved], _LIMIT)
        print(f"SMA patch: an even share of every category, {len(keep)} of {len(all_test_entries_involved)} questions", file=sys.stderr, flush=True)
        cases = original(args, model_name, all_test_categories, all_test_file_paths, all_test_entries_involved)
        return [case for case in cases if case["id"] in keep]

    module.collect_test_cases = collect_test_cases


def _aligned(runner, with_answers):
    def run(handler, model_result, prompt, *rest, **kwargs):
        ids = {entry["id"] for entry in model_result}
        prompt = [entry for entry in prompt if entry["id"] in ids]
        if with_answers:
            rest = ([entry for entry in rest[0] if entry["id"] in ids], *rest[1:])
        return runner(handler, model_result, prompt, *rest, **kwargs)

    return run


def _patch_evaluate(module):
    module.multi_turn_runner = _aligned(module.multi_turn_runner, True)
    module.ast_file_runner = _aligned(module.ast_file_runner, True)
    module.relevance_file_runner = _aligned(module.relevance_file_runner, False)
    module.executable_file_runner = _aligned(module.executable_file_runner, False)


_PATCHES = {"bfcl._llm_response_generation": _patch_generate, "bfcl.eval_checker.eval_runner": _patch_evaluate}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        patch = _PATCHES.get(name)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def _run_hidden_sitecustomize():
    """Run the sitecustomize this file hides (the image's own, if any), as Python would have."""
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    for entry in sys.path:
        if os.path.abspath(entry or ".") == here:
            continue
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", [entry])
        if spec is not None and spec.loader is not None:
            spec.loader.exec_module(importlib.util.module_from_spec(spec))
            return


if _LIMIT:
    sys.meta_path.insert(0, _Finder())
_run_hidden_sitecustomize()
