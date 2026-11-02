"""Grade a LiveCodeBench run's saved code again, inside NVIDIA's eval-factory container (`sma rescore`).

For runs graded before SMA's patch (sitecustomize.py here), whose runner gave each test the request
timeout instead of LiveCodeBench's 6 s. It reads /results/*_eval_all.json, grades each problem's saved
code with LiveCodeBench's own grader at --timeout seconds per test, and writes the same file to /out with
graded_list, pass@1, and metadata replaced as the runner would set them. No model is asked anything.
"""

import argparse
import glob
import json
import os

from livecodebench.benchmarks.code_generation import load_code_generation_dataset
from livecodebench.evaluation.compute_code_generation_metrics import codegen_metrics
from livecodebench.evaluation.pass_k_utils import extract_instance_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--processes", type=int, default=8)
    args = parser.parse_args()

    found = sorted(glob.glob("/results/*_eval_all.json"))
    if len(found) != 1:
        raise SystemExit(f"Expected one *_eval_all.json in the attempt, found {len(found)}.")
    rows = json.load(open(found[0], encoding="utf-8"))
    problems = {
        problem.question_id: problem
        for problem in load_code_generation_dataset(args.release, start_date=args.start_date, end_date=args.end_date)
    }
    missing = [row["question_id"] for row in rows if row["question_id"] not in problems]
    if missing:
        raise SystemExit(f"{len(missing)} problems are not in {args.release} between the dates given, e.g. {missing[:3]}.")
    samples = [problems[row["question_id"]].get_evaluation_sample() for row in rows]
    from sma_grading import grading_turn  # Beside this script: the grading lock and CPUs, as in a run.

    with grading_turn():
        metrics = codegen_metrics(
            samples, [row["code_list"] for row in rows], k_list=[1], num_process_evaluate=args.processes, timeout=args.timeout
        )
    graded = extract_instance_results(metrics[1])
    for row, graded_list, metadata in zip(rows, graded, metrics[2]):
        row["graded_list"] = graded_list
        row["pass@1"] = graded_list.count(True) / len(graded_list)
        row["metadata"] = metadata
    with open(os.path.join("/out", os.path.basename(found[0])), "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=4)
    passed = sum(1 for graded_list in graded if all(graded_list))
    print(f"Regraded {len(rows)} problems at {args.timeout} s per test: {passed} pass.")


if __name__ == "__main__":
    main()
