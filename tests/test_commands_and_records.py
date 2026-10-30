import json
import socket
from pathlib import Path

import pytest

from sma.commands import bench_argv, bench_cells, serve_argv
from sma.config import default_config, load_config
from sma.errors import SmaError
from sma.identity import assert_alias_safe
from sma.records import new_manifest, publish
from sma.schemautil import validate_instance
from sma.scrub import scrub_text
from sma.speedmath import decode_tok_s, prefill_tok_s
from tests.harness import SERVE_PLAN


def test_serve_command_pins_the_contract() -> None:
    cfg = default_config()
    bf16 = serve_argv(
        cfg,
        mode="quality",
        served_name="org/model",
        is_moe=False,
        quant_flag="bf16",
        moe_backend=None,
        batch_invariant=True,
        plan=SERVE_PLAN,
    )
    assert bf16[0:2] == ["vllm", "serve"]
    assert "--tensor-parallel-size" in bf16
    assert bf16[bf16.index("--tensor-parallel-size") + 1] == "1"
    assert "--no-enable-flashinfer-autotune" in bf16
    assert "--no-async-scheduling" in bf16
    assert "--moe-backend" not in bf16
    assert "--max-logprobs" in bf16
    speed = serve_argv(
        cfg,
        mode="speed",
        served_name="org/model",
        is_moe=True,
        quant_flag="bf16",
        moe_backend="flashinfer_b12x",
        batch_invariant=False,
        plan=SERVE_PLAN,
    )
    assert "--no-enable-flashinfer-autotune" not in speed
    assert "--moe-backend" not in speed
    nvfp4 = serve_argv(
        cfg,
        mode="quality",
        served_name="org/model",
        is_moe=True,
        quant_flag="nvfp4",
        moe_backend="flashinfer_b12x",
        batch_invariant=True,
        plan=SERVE_PLAN,
    )
    assert nvfp4[nvfp4.index("--moe-backend") + 1] == "flashinfer_b12x"
    dense = serve_argv(
        cfg,
        mode="quality",
        served_name="org/model",
        is_moe=False,
        quant_flag="nvfp4",
        moe_backend=None,
        batch_invariant=True,
        plan=SERVE_PLAN,
    )
    # vLLM 0.21 has no --linear-backend flag: it refused it on a Spark. The GEMM is an env var.
    assert "--linear-backend" not in dense
    from sma.commands import serve_env

    env = serve_env(cfg, mode="quality", batch_invariant=True, is_moe=False, quant_flag="nvfp4")
    assert env["VLLM_NVFP4_GEMM_BACKEND"] == "flashinfer-b12x"
    assert "VLLM_NVFP4_GEMM_BACKEND" not in serve_env(cfg, mode="quality", batch_invariant=True, is_moe=True, quant_flag="nvfp4")
    assert "VLLM_NVFP4_GEMM_BACKEND" not in serve_env(cfg, mode="quality", batch_invariant=True, quant_flag="bf16")


def _serve(engine: str, mode: str = "quality") -> list[str]:
    return serve_argv(
        default_config(),
        mode=mode,
        served_name="org/model",
        is_moe=False,
        quant_flag="gguf" if engine == "llamacpp" else "bf16",
        moe_backend=None,
        batch_invariant=False,
        plan=SERVE_PLAN,
        engine=engine,
    )


@pytest.mark.parametrize("mode", ["quality", "speed"])
def test_every_engine_runs_an_8_bit_kv_cache_sized_by_the_memory_plan(mode: str) -> None:
    vllm = _serve("vllm", mode)
    assert vllm[vllm.index("--kv-cache-dtype") + 1] == "fp8"
    assert vllm[vllm.index("--kv-cache-memory-bytes") + 1] == "20000000000"
    assert vllm[vllm.index("--max-model-len") + 1] == "8192"
    sglang = _serve("sglang", mode)
    assert sglang[sglang.index("--kv-cache-dtype") + 1] == "fp8_e4m3"
    assert sglang[sglang.index("--max-total-tokens") + 1] == "348768"
    assert sglang[sglang.index("--context-length") + 1] == "8192"
    llama = _serve("llamacpp", mode)
    assert (llama[llama.index("-ctk") + 1], llama[llama.index("-ctv") + 1]) == ("q8_0", "q8_0")
    assert llama[llama.index("-fa") + 1] == "on"
    assert llama[llama.index("-c") + 1] == "348768"
    assert llama[llama.index("--kv-unified-per-slot") + 1] == "8192"
    assert llama[llama.index("-m") + 1] == "/model/Tiny-UD-Q4_K_XL.gguf"
    for argv in (vllm, sglang, llama):
        joined = " ".join(argv)
        assert "f16" not in joined.replace("--no-enable-flashinfer-autotune", "") and "bfloat16" not in joined


def test_sglang_turns_deep_gemm_off_on_the_gb10() -> None:
    from sma.commands import serve_env

    cfg = default_config()
    for mode in ("quality", "speed"):
        assert serve_env(cfg, mode=mode, batch_invariant=False, engine="sglang")["SGLANG_ENABLE_JIT_DEEPGEMM"] == "0"


@pytest.mark.parametrize("key", ["max_model_len", "kv_cache_gib"])
def test_removed_context_keys_say_what_replaced_them(key: str) -> None:
    from sma.config import config_from_mapping

    with pytest.raises(SmaError, match=f"config.engine.{key} is gone. The memory plan"):
        config_from_mapping({"engine": {key: 8}})


def test_bench_command_and_derived_rates() -> None:
    cfg = default_config()
    cell = next(item for item in bench_cells(cfg) if item.name.startswith("prefill_c1"))
    argv = bench_argv(cfg, container="sma-run-speed", served_name="org/model", cell=cell)
    text = " ".join(argv)
    assert "vllm bench serve" in text
    assert "--percentile-metrics ttft,tpot,itl,e2el" in text
    assert "--metric-percentiles 50,90,99" in text
    assert "--save-result" in text
    assert "--max-concurrency 1" in text
    assert prefill_tok_s(2048, 100.0) == pytest.approx(20480.0)
    assert decode_tok_s(20.0) == pytest.approx(50.0)
    assert prefill_tok_s(2048, 0) is None
    assert decode_tok_s(None) is None


def test_blank_manifest_matches_the_schema(tmp_path: Path) -> None:
    cfg = default_config()
    cfg.runs_dir = str(tmp_path / "runs")
    manifest = new_manifest(
        cfg=cfg,
        repo="example/tiny",
        quant_flag="bf16",
        revision=None,
        run_id="20260101T000000Z-abcd1234",
        reference_run_id=None,
        pair_job_id=None,
    )
    run_dir = tmp_path / "runs" / "example_tiny" / "vllm" / "bf16" / manifest["run_id"]
    run_dir.mkdir(parents=True)
    summary = publish(cfg, run_dir, manifest)
    validate_instance(json.loads((run_dir / "manifest.json").read_text()), "manifest.schema.json")
    validate_instance(summary, "summary.schema.json")
    assert summary["perplexity"] is None
    assert summary["speed"]["prefill_tok_s"] is None
    assert summary["engine"] == "vllm"
    assert summary["energy"]["joules"] is None


def test_alias_rejects_a_hostname(monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "gpu-host-1")
    monkeypatch.setattr(socket, "getfqdn", lambda: "gpu-host-1")
    with pytest.raises(SmaError, match="hostname"):
        assert_alias_safe("gpu-host-1")
    with pytest.raises(SmaError, match="dot"):
        assert_alias_safe("spark.lab")


def test_scrub_removes_hostname_and_token(monkeypatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "secret-host")
    monkeypatch.setattr(socket, "getfqdn", lambda: "secret-host")
    monkeypatch.setenv("HF_TOKEN", "hf_test_token_value")
    text = scrub_text("failed on secret-host token hf_test_token_value")
    assert "secret-host" not in text
    assert "hf_test_token_value" not in text
    assert "<host>" in text
    assert "<redacted>" in text


def test_scrub_removes_a_hostname_that_ends_a_sentence_but_keeps_longer_names(monkeypatch) -> None:
    # nemo-evaluator logged "parent directories lack a+x: /home/<user>." on a Spark whose user name is its hostname.
    monkeypatch.setattr(socket, "gethostname", lambda: "spark1")
    monkeypatch.setattr(socket, "getfqdn", lambda *_args: "spark1")

    assert scrub_text("lack a+x: /home/spark1. Grant") == "lack a+x: /home/<host>. Grant"
    assert scrub_text("on spark1.") == "on <host>."
    # Another machine's name that starts with this one is not this hostname.
    assert scrub_text("spark1.example.org and spark10 and spark1-b") == "spark1.example.org and spark10 and spark1-b"


def test_repo_config_loads() -> None:
    cfg = load_config(Path(__file__).resolve().parents[1] / "sma.yaml")
    assert cfg.engine.name == "vllm"
    assert cfg.engine.image == "nvcr.io/nvidia/vllm:26.05.post1-py3"
    assert cfg.engine.sglang_image == "lmsysorg/sglang:v0.5.10.post1-cu130"
    assert cfg.engine.sglang_attention_backend == "flashinfer"
    assert cfg.logprobs.corpus_version == "sma-corpus-v3"
    assert cfg.logprobs.num_windows == 48
    assert cfg.telemetry.sw_power_cap_invalidates is False
    assert cfg.engine.moe_backend == "flashinfer_b12x"
    assert cfg.determinism.batch_sizes == [1, 8, 64]
    assert cfg.logprobs.top_k == 64
    assert cfg.host_alias == "gpu-host-1"
    # Nested sections load as their dataclasses, not plain mappings.
    assert (cfg.kernelbench.level, cfg.kernelbench.enabled) == (1, True)
    assert (cfg.swebench.workers, cfg.swebench.step_limit) == (8, 250)
    assert (cfg.browsecomp.limit, cfg.browsecomp.judge_model, cfg.browsecomp.judge_concurrency) == (100, "ornith-1.5-122b", 1)
    assert cfg.browsecomp.judge_extra_body == {"chat_template_kwargs": {"enable_thinking": False}, "cache_prompt": False}
    assert cfg.browsecomp == default_config().browsecomp
    assert [suite.name for suite in cfg.quality.suites] == ["ifeval", "gsm8k", "mmlu_pro", "gpqa_diamond", "musique", "bfcl_v3", "livecodebench"]
    # Owner decision 2026-10-29: LiveCodeBench in place of SWE-bench, in LiveCodeBench's leaderboard window.
    livecodebench = cfg.quality.suites[-1]
    assert (livecodebench.task, livecodebench.sampling, livecodebench.questions) == ("codegeneration_release_v6", True, 454)
    assert livecodebench.params == {"extra": {"n_samples": 1, "start_date": "2024-08-01", "end_date": "2025-05-01"}}
    assert cfg.swebench.enabled is False
    assert next(suite for suite in cfg.quality.suites if suite.name == "musique").metric == "answer_f1"
    assert [suite.name for suite in cfg.quality.suites] == [suite.name for suite in default_config().quality.suites]


def test_every_default_quality_suite_names_a_benchmark_the_installed_nel_has() -> None:
    pytest.importorskip("nemo_evaluator", reason="the quality extra is not installed")
    from lm_eval.tasks import TaskManager
    from nemo_evaluator.environments.registry import list_environments

    from sma.commands import packaged_bench

    builtins = set(list_environments())
    lm_eval_tasks = set(TaskManager().all_tasks)
    missing = []
    for suite in default_config().quality.suites:
        bench = suite.bench or suite.name
        if suite.image is not None or packaged_bench(bench) is not None:
            continue
        if bench.startswith("lm-eval://"):
            if bench[len("lm-eval://") :] not in lm_eval_tasks:
                missing.append(bench)
        elif bench not in builtins:
            missing.append(bench)
    assert missing == []


def test_the_nel_config_targets_the_chat_endpoint_in_the_runs_thinking_mode_with_no_answer_limit() -> None:
    from sma.commands import nel_argv, nel_config
    from sma.config import QualitySuite, for_thinking

    cfg = for_thinking(default_config(), "off")
    config = nel_config(
        cfg, served_name="org/model", base_url="http://127.0.0.1:8000/", suite=QualitySuite("gsm8k", 50, bench="sma:gsm8k")
    )

    service = config["services"]["sma"]
    assert service["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert (service["model"], service["protocol"]) == ("org/model", "chat_completions")
    # Thinking off: greedy. No max_tokens: an answer may use the rest of its stream's context (owner decision
    # 2026-10-29; 4,096 cut off 30% of LFM3-1.7B's MMLU-Pro answers).
    assert service["generation"] == {"temperature": 0.0, "seed": 0}
    assert service["proxy"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert service["proxy"]["interceptors"] == [{"name": "reasoning"}]
    assert (service["proxy"]["max_retries"], service["proxy"]["request_timeout"]) == (2, 14400.0)
    [bench] = config["benchmarks"]
    # The packaged grader is copied next to nel.yaml, so the config holds no absolute path.
    assert bench["name"] == "gsm8k.py"
    assert (bench["max_problems"], bench["max_concurrent"], bench["shuffle_seed"]) == (50, 8, 0)
    assert nel_argv(cfg, output_name="attempt-2") == ["nel", "eval", "run", "nel.yaml", "-o", "attempt-2"]

    # Thinking on: no temperature, so the engine applies the model's own sampling.
    on = nel_config(for_thinking(default_config(), "on"), served_name="m", base_url="http://x", suite=QualitySuite("gsm8k", 50))
    assert on["services"]["sma"]["generation"] == {"seed": 0}
    assert on["services"]["sma"]["proxy"]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    capped = default_config()
    capped.quality.max_tokens = 4096
    assert nel_config(capped, served_name="m", base_url="http://x", suite=QualitySuite("gsm8k", 50))["services"]["sma"]["generation"]["max_tokens"] == 4096

    cfg = default_config()
    plain = nel_config(cfg, served_name="m", base_url="http://127.0.0.1:8000", suite=QualitySuite("mmlu_pro", None))
    assert "extra_body" not in plain["services"]["sma"]["proxy"]
    assert (plain["benchmarks"][0]["name"], plain["benchmarks"][0]["max_problems"]) == ("mmlu_pro", None)


def test_the_installed_nel_accepts_the_config_sma_writes() -> None:
    pytest.importorskip("nemo_evaluator", reason="the quality extra is not installed")
    from nemo_evaluator.config.eval_config import parse_eval_config

    from sma.commands import nel_config

    cfg = default_config()
    for suite in cfg.quality.suites:
        parsed = parse_eval_config(nel_config(cfg, served_name="m", base_url="http://127.0.0.1:8000", suite=suite))
        assert parsed.benchmarks[0].max_concurrent == cfg.quality.concurrency


def test_an_unknown_packaged_benchmark_is_refused_before_any_download() -> None:
    from sma.config import config_from_mapping, validate_config

    with pytest.raises(SmaError, match="sma:nope is not a benchmark SMA ships. Known: sma:gsm8k"):
        validate_config(config_from_mapping({"quality": {"suites": [{"name": "x", "bench": "sma:nope"}]}}))


def test_quality_max_tokens_must_leave_room_for_the_prompt() -> None:
    from sma.config import config_from_mapping, validate_config

    with pytest.raises(SmaError, match="quality.max_tokens"):
        validate_config(config_from_mapping({"quality": {"max_tokens": 8192}}))


def test_scrub_keeps_a_per_token_count_and_still_redacts_token_fields() -> None:
    from sma.scrub import scrub_obj

    cleaned = scrub_obj({"kv_bytes_per_token": 57344, "auth_token": "x", "token": "y", "hf_token": "z"})
    assert cleaned == {"kv_bytes_per_token": 57344, "auth_token": "<redacted>", "token": "<redacted>", "hf_token": "<redacted>"}


def test_the_thinking_mode_is_set_by_the_run_not_the_config() -> None:
    from sma.config import config_from_mapping, for_thinking, validate_config

    with pytest.raises(SmaError, match="--thinking on"):
        validate_config(config_from_mapping({"quality": {"chat_template_kwargs": {"enable_thinking": False}}}))
    with pytest.raises(SmaError, match="Thinking mode must be one of off, on"):
        for_thinking(default_config(), "maybe")
