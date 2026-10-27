"""Shared config and service wiring for pipeline tests."""

from __future__ import annotations

import socket
from pathlib import Path

from sma.config import QualitySuite, default_config, validate_config
from sma.pipeline import Hooks, Services
from tests.fakes import FakeBencher, FakeEvaluator, FakeRuntime, FixtureFetcher, telemetry_factory, write_snapshot


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def make_config(tmp_path: Path, *, port: int | None = None):
    cfg = default_config()
    cfg.runs_dir = str(tmp_path / "runs")
    cfg.lock_path = str(tmp_path / "gpu.lock")
    cfg.queue.dir = str(tmp_path / "queue")
    cfg.hf.cache_dir = str(tmp_path / "hf")
    cfg.engine.port = port or free_port()
    cfg.retries.attempts = 1
    cfg.retries.backoff_s = 0
    cfg.timeouts.server_start_s = 8
    cfg.timeouts.health_poll_s = 0.05
    cfg.timeouts.health_request_s = 2
    cfg.speed.warmup_requests = 0
    cfg.speed.warmup_soak_s = 0
    cfg.speed.num_prompts = 4
    cfg.speed.sweep_concurrency = [1, 4, 8]
    cfg.telemetry.idle_timeout_s = 0
    cfg.telemetry.poll_s = 0
    cfg.telemetry.drop_page_cache = False
    cfg.logprobs.window_tokens = 8
    cfg.logprobs.num_windows = 2
    cfg.logprobs.top_k = 4
    cfg.determinism.prompts = ["hello"]
    cfg.determinism.batch_sizes = [1, 8]
    cfg.quality.suites = [QualitySuite("ifeval", 2, bench="lm-eval://ifeval")]
    cfg.quality.concurrency = 2
    cfg.quality.max_tokens = 16
    cfg.engine.max_model_len = 64
    validate_config(cfg)
    return cfg


def make_services(tmp_path: Path, cfg, *, quant: str = "bf16", moe: bool = False, model_type: str = "llama"):
    snapshot = write_snapshot(tmp_path / "snapshot", quant=quant, moe=moe, model_type=model_type)
    fetcher = FixtureFetcher(snapshot, repo="example/tiny", quant_flag=quant)
    runtime = FakeRuntime()
    hooks = Hooks()

    def _encode(_text: str) -> list[int]:
        width = cfg.logprobs.window_tokens * cfg.logprobs.num_windows + 8
        return [(index % 40) + 1 for index in range(width)]

    services = Services(
        fetcher=fetcher,
        runtime=runtime,
        evaluator=FakeEvaluator(),
        bencher=FakeBencher(),
        telemetry_factory=telemetry_factory,
        encode=_encode,
        hooks=hooks,
    )
    return services, fetcher, runtime
