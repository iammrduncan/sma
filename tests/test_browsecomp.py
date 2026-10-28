"""BrowseComp-Plus (owner decision 2026-10-28): its protocol, its scoring, and a whole run on fakes.

The search server here is a real HTTP server on this machine's loopback, standing in for the runner's
end of SMA's SSH forward. The runner and the judge are fakes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from sma import browsecomp as bcp
from sma.errors import SmaError
from sma.httpapi import ApiError
from sma.pipeline import RunSpec, execute
from sma.records import LARGE_FILES
from sma.schemautil import validate_instance
from tests.harness import free_port, make_config, make_services

QWEN3_TEMPLATE = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{%- endif %}{%- endif %}"
)
QUESTIONS = {
    "1": ("Which city hosts the tower built for the 1889 fair?", "Paris"),
    "2": ("Which city was founded on seven hills?", "Rome"),
    "10": ("Which city has the Louvre?", "Paris"),
}
QRELS_EVIDENCE = "1 Q0 11 1\n1 Q0 13 1\n2 Q0 12 1\n10 Q0 99 1\n"
QRELS_GOLDS = "1 Q0 11 1\n2 Q0 12 1\n10 Q0 99 1\n"


# The protocol


def test_the_prompts_are_byte_for_byte_browsecomp_plus_own() -> None:
    # Hashes of the texts at 04694903 (search_agent/prompts.py, scripts_evaluation/evaluate_run.py).
    # The grader has a trailing space after "[response]." that editors like to strip.
    assert hashlib.sha256(bcp.QUERY_TEMPLATE.encode()).hexdigest() == "4a24b6a8b8186eef688cdb4dffa50e305f206546f8f2c6e95d6bc90e48ec53ad"
    assert hashlib.sha256(bcp.GRADER_TEMPLATE.encode()).hexdigest() == "71fb29f51a56cea9fe647331d525648474195fa4816f7d05632459d572419d90"
    assert "extracted from the [response]. \n" in bcp.GRADER_TEMPLATE
    assert bcp.search_tool(5)["function"]["description"].startswith("Perform a search on a knowledge source. Returns top-5 hits")


@pytest.mark.parametrize(
    ("reply", "correct", "confidence", "parse_error"),
    [
        ("extracted_final_answer: Paris\nreasoning: same\ncorrect: yes\nconfidence: 90%", True, 90.0, False),
        ("**extracted_final_answer:** Rome\n**correct:** no\n**confidence:** 150", False, 100.0, False),
        # A real reply from the judge test: no correct line, so it counts as wrong, as in BrowseComp-Plus.
        ("extracted_final_answer: Stephan Weyte\n\ncorrect_answer: Stephan Weyte\n\nconfidence: 90%", None, 90.0, True),
        ("", None, None, True),
    ],
)
def test_the_judges_reply_is_read_as_browsecomp_plus_reads_it(reply: str, correct: bool | None, confidence: float | None, parse_error: bool) -> None:
    parsed = bcp.parse_judge(reply)
    assert (parsed["correct"], parsed["confidence"], parsed["parse_error"]) == (correct, confidence, parse_error)


def test_the_same_questions_are_picked_whatever_order_the_ids_come_in() -> None:
    ids = [str(n) for n in (3, 1, 20, 2, 10)]
    picked = bcp.pick_queries(ids, 3, seed=0)
    assert picked == bcp.pick_queries(list(reversed(ids)), 3, seed=0)
    assert len(picked) == 3 and set(picked) <= set(ids)
    assert bcp.pick_queries(ids, None, seed=0)[:3] == picked


def _reply(message: dict[str, Any], finish: str, tokens: int = 10) -> dict[str, Any]:
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": {"prompt_tokens": 5, "completion_tokens": tokens}}


def _call(arguments: str, name: str = "search") -> dict[str, Any]:
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": arguments}}


def test_a_question_ends_on_the_models_own_answer_and_keeps_reasoning_out_of_the_history() -> None:
    sent: list[dict[str, Any]] = []
    replies = iter([
        _reply({"role": "assistant", "content": None, "reasoning": "think", "tool_calls": [_call('{"query": "q"}')]}, "tool_calls"),
        _reply({"role": "assistant", "content": "Exact Answer: X", "reasoning_content": "done"}, "stop"),
    ])  # fmt: skip

    def chat(body: dict[str, Any]) -> dict[str, Any]:
        sent.append(json.loads(json.dumps(body)))
        return next(replies)

    record = bcp.run_query("Q?", chat=chat, search=lambda q: '[{"docid": "7", "score": 1.0, "snippet": "s"}]', model="m",
                           k=5, max_tokens=100, max_iterations=10, enable_thinking=True)  # fmt: skip
    assert record["status"] == "completed" and record["turns"] == 2
    assert record["tool_call_counts"] == {"search": 1} and record["retrieved_docids"] == ["7"]
    assert bcp.final_response(record) == "Exact Answer: X"
    assert [item["type"] for item in record["result"]] == ["reasoning", "tool_call", "reasoning", "output_text"]
    second = sent[1]
    assert second["max_tokens"] == 90  # one budget for the whole conversation
    assert second["chat_template_kwargs"] == {"enable_thinking": True}
    assert not any(key in second for key in ("temperature", "top_p", "top_k"))
    assistant = second["messages"][1]
    assert "reasoning" not in assistant and assistant["tool_calls"][0]["id"] == "c1"
    assert second["messages"][2]["role"] == "tool" and second["messages"][2]["tool_call_id"] == "c1"


def test_bad_arguments_go_back_to_the_model_and_a_spent_budget_ends_the_question() -> None:
    calls = [_reply({"role": "assistant", "content": None, "tool_calls": [_call("{not json")]}, "tool_calls", tokens=60)] * 3
    replies = iter(calls)
    record = bcp.run_query("Q?", chat=lambda body: next(replies), search=lambda q: "[]", model="m", k=5,
                           max_tokens=100, max_iterations=10, enable_thinking=None)  # fmt: skip
    # 60 + 60 tokens spend the 100-token budget after two replies; the run never answered.
    assert record["turns"] == 2 and record["status"] == "tool_calls" and bcp.final_response(record) is None
    assert record["result"][0]["output"].startswith("Error executing search:")


def test_a_refused_request_ends_the_question_as_wrong_but_a_dead_server_stops_the_step() -> None:
    def refused(body: dict[str, Any]) -> dict[str, Any]:
        raise SmaError("http://x/v1/chat/completions returned 400: maximum context length")

    record = bcp.run_query("Q?", chat=refused, search=lambda q: "[]", model="m", k=5, max_tokens=100, max_iterations=10, enable_thinking=None)
    assert record["status"] == "error" and "maximum context length" in record["error"]

    def down(body: dict[str, Any]) -> dict[str, Any]:
        raise ApiError("timed out")

    with pytest.raises(ApiError):
        bcp.run_query("Q?", chat=down, search=lambda q: "[]", model="m", k=5, max_tokens=100, max_iterations=10, enable_thinking=None)


def test_the_score_counts_every_picked_question_and_leaves_calibration_null() -> None:
    evidence, golds = bcp.read_qrels(QRELS_EVIDENCE), bcp.read_qrels(QRELS_GOLDS)
    finished = {"status": "completed", "result": [{"type": "output_text", "output": "x"}], "tool_call_counts": {"search": 2},
                "retrieved_docids": ["11", "12"], "usage": {"completion_tokens": 50}, "turns": 3}  # fmt: skip
    unfinished = {"status": "tool_calls", "result": [], "tool_call_counts": {"search": 4}, "retrieved_docids": [], "usage": {}, "turns": 4}
    rows = [
        bcp.per_query("1", finished, {"parsed": {"correct": True, "parse_error": False, "confidence": 90.0}}, evidence, golds),
        bcp.per_query("2", finished, {"parsed": {"correct": None, "parse_error": True, "confidence": None}}, evidence, golds),
        bcp.per_query("10", unfinished, {"skipped": "no final answer"}, evidence, golds),
    ]
    assert [row["evidence_recall"] for row in rows] == [0.5, 1.0, 0.0]
    scores = bcp.score(rows, judge="j", limit=3)
    assert (scores["primary_score"], scores["n"], scores["correct"]) == (pytest.approx(1 / 3), 3, 1)
    assert scores["judge_parse_errors"] == 1 and scores["calibration_error"] is None
    assert scores["search_calls"] == pytest.approx(8 / 3) and scores["statuses"] == {"completed": 2, "tool_calls": 1}


# A whole run


class SearchServer:
    """What the runner's search_server.py answers, on a free local port."""

    def __init__(self) -> None:
        self.port = free_port()
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args: object) -> None:
                return

            def _send(self, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                self._send({"ok": True, "docs": 3})

            def do_POST(self) -> None:  # noqa: N802
                owner.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                hits = [{"docid": "11", "score": 2.5, "snippet": "Paris has the tower."}, {"docid": "12", "score": 1.5, "snippet": "Rome."}]
                self._send({"output": json.dumps(hits, indent=2)})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeForward:
    class process:  # noqa: N801 - the shape of proc.Following
        @staticmethod
        def poll() -> None:
            return None

    def stop(self, timeout_s: float = 10.0) -> None:
        return None


class FakeRunner:
    def __init__(self) -> None:
        self.files = {
            "data/queries.jsonl": "".join(json.dumps({"query_id": q, "query": text, "answer": a}) + "\n" for q, (text, a) in QUESTIONS.items()),
            "data/qrel_evidence.txt": QRELS_EVIDENCE,
            "data/qrel_golds.txt": QRELS_GOLDS,
        }
        self.started: list[str] = []
        self.stopped = False

    def shell(self, script: str, *, what: str, timeout_s: float = 300.0) -> str:
        return ""

    def read_text(self, rel: str) -> str | None:
        return self.files.get(rel)

    def start(self, rel_dir: str, name: str, command: str) -> None:
        self.started.append(command)

    def status(self, rel_dir: str, name: str) -> tuple[int | None, bool]:
        return (0, False) if self.stopped else (None, True)

    def stop(self, run_id: str) -> None:
        self.stopped = True

    def forward(self, local_port: int, remote_port: int, log_path: Path) -> FakeForward:
        return FakeForward()


class FakeJudge:
    def __init__(self) -> None:
        self.asked: list[dict[str, Any]] = []

    def ask(self, prompt: str, *, max_tokens: int, temperature: float | None, extra_body: dict[str, Any] | None = None) -> dict[str, Any]:
        self.asked.append({"prompt": prompt, "extra_body": extra_body, "temperature": temperature})
        right = "[correct_answer]: Paris" in prompt
        return {"model": "judge", "content": f"extracted_final_answer: Paris\ncorrect: {'yes' if right else 'no'}\nconfidence: 90%",
                "reasoning": None, "finish_reason": "stop", "usage": {}}  # fmt: skip


def _setup(tmp_path: Path, *, tool_parser: str | None = "hermes"):
    cfg = make_config(tmp_path)
    cfg.browsecomp.enabled = True
    cfg.browsecomp.limit = 2
    cfg.browsecomp.workers = 2
    cfg.engine.tool_call_parser = tool_parser
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    (tmp_path / "snapshot" / "tokenizer_config.json").write_text(json.dumps({"chat_template": QWEN3_TEMPLATE}))
    return cfg, services


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_run_answers_with_search_on_the_runner_grades_with_the_judge_and_keeps_the_text_out_of_git(tmp_path: Path, monkeypatch) -> None:
    import tests.fake_vllm as fake_vllm

    seen: list[list[dict[str, Any]]] = []
    real = fake_vllm.FakeVLLM.start

    def start(self):
        seen.append(self.chat_requests)  # type: ignore[arg-type]
        return real(self)

    monkeypatch.setattr(fake_vllm.FakeVLLM, "start", start)
    cfg, services = _setup(tmp_path)
    search = SearchServer()
    cfg.browsecomp.local_port = search.port
    runner, judge = FakeRunner(), FakeJudge()
    services.research_runner, services.judge = runner, judge
    try:
        run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    finally:
        search.stop()
    run_dir = next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))
    suite = run_dir / "quality" / bcp.SUITE

    settings = _json(suite / "settings.json")
    picked = settings["query_ids"]
    assert len(picked) == 2 and picked == bcp.pick_queries(list(QUESTIONS), 2, cfg.seed)
    assert settings["thinking"]["enable_thinking"] is True
    assert runner.started == [f".venv/bin/python search_server.py --index indexes/bm25 --tokenizer data/tokenizer --port {cfg.browsecomp.search_port} --snippet-max-tokens 512 --tag runs/{run_id}"]
    assert runner.stopped and [r["k"] for r in search.requests] == [5, 5]

    requests = [request for batch in seen for request in batch]
    assert len(requests) == 4  # a search, then an answer, for each question
    assert all(r["tools"][0]["function"]["name"] == "search" and r["chat_template_kwargs"] == {"enable_thinking": True} for r in requests)
    assert not any(key in requests[0] for key in ("temperature", "top_p", "top_k"))

    # The judge saw each question and its answer, with thinking off, no prompt cache, at temperature 0.
    assert len(judge.asked) == 2
    want_body = {"chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    assert all(a["extra_body"] == want_body and a["temperature"] == 0.0 for a in judge.asked)
    want_right = sum(1 for q in picked if QUESTIONS[q][1] == "Paris")
    scores = _json(suite / "scores.json")
    assert (scores["primary_metric"], scores["n"], scores["correct"]) == ("accuracy", 2, want_right)
    assert scores["primary_score"] == pytest.approx(want_right / 2)
    assert scores["judge"] == "ornith-1.5-122b" and scores["context_per_stream"] is not None
    evidence = bcp.read_qrels(QRELS_EVIDENCE)
    assert scores["evidence_recall"] == pytest.approx(sum(len({"11", "12"} & set(evidence[q])) / len(evidence[q]) for q in picked) / 2)

    # Questions and answers are only in the git-ignored bundle; the committed files have numbers only.
    assert not (suite / "queries").exists()
    committed = "".join(path.read_text(encoding="utf-8") for path in suite.iterdir() if path.suffix in (".json", ".jsonl"))
    for text, answer in QUESTIONS.values():
        assert text not in committed and answer not in committed
    with gzip.open(suite / "trajectories.jsonl.gz", "rt", encoding="utf-8") as handle:
        bundle = [json.loads(line) for line in handle]
    assert sorted(row["query_id"] for row in bundle) == sorted(picked) and "Paris" in json.dumps(bundle)
    manifest = _json(run_dir / "manifest.json")
    assert "quality/browsecomp_plus/trajectories.jsonl.gz" in manifest["large_files"]
    assert "quality/browsecomp_plus/trajectories.jsonl.gz" in LARGE_FILES
    assert manifest["steps"]["browsecomp_plus"]["status"] == "done" and manifest["steps"]["browsecomp_plus_results"]["status"] == "done"
    summary = _json(run_dir / "summary.json")
    assert summary["quality"]["browsecomp_plus"]["score"] == pytest.approx(want_right / 2)
    validate_instance(summary, "summary.schema.json")
    validate_instance(manifest, "manifest.schema.json")


def test_without_a_judge_or_a_tool_parser_the_score_is_null_and_says_why(tmp_path: Path, monkeypatch) -> None:
    for name in ("SMA_JUDGE_URL", "SMA_JUDGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg, services = _setup(tmp_path)
    services.research_runner = FakeRunner()
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    summary = _json(next((tmp_path / "runs").glob(f"*/*/*/{run_id}")) / "summary.json")
    assert summary["quality"]["browsecomp_plus"]["score"] is None
    assert "SMA_JUDGE_URL" in summary["quality"]["browsecomp_plus"]["error"]

    cfg, services = _setup(tmp_path / "second", tool_parser=None)
    services.research_runner, services.judge = FakeRunner(), FakeJudge()
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    summary = _json(next((tmp_path / "second" / "runs").glob(f"*/*/*/{run_id}")) / "summary.json")
    assert "--tool-parser" in summary["quality"]["browsecomp_plus"]["error"]


def test_the_runner_pins_the_same_files_as_sma_and_every_package() -> None:
    import re

    folder = Path(__file__).resolve().parents[1] / "services" / "browsecomp-plus"
    prepare = (folder / "prepare.py").read_text(encoding="utf-8")
    for name in ("COMMIT", "QUERIES_REVISION", "INDEX_REVISION", "CANARY"):
        assert f'{name} = "{getattr(bcp, name)}"' in prepare
    lines = (folder / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pins = [line.split(" ")[0] for line in lines if line and not line.startswith((" ", "#"))]
    assert pins and all(re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[^=\s]+", pin) for pin in pins)
    assert all(line.rstrip().endswith("\\") or line.startswith((" ", "#")) or not line for line in lines)
    assert {"pyserini==1.2.0", "torch==2.7.0+cpu", "openai==1.97.0", "transformers==4.53.2"} <= set(pins)
    server = (folder / "search_server.py").read_text(encoding="utf-8")
    assert '"127.0.0.1"' in server and "def log_message" in server  # loopback only, and no query text in its log
