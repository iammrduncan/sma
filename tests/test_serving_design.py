"""The serving design (docs/serving-design.md): engines by format, an 8-bit KV cache, the memory plan,
and deployment notes. Pipeline tests run on fakes; nothing here needs a GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sma.config import config_from_mapping, default_config, validate_config
from sma.engines import engine_name
from sma.errors import SmaError
from sma.kl import parse_perplexity_log
from sma.memplan import KvLayout
from sma.pipeline import RunSpec, SimulatedKill, execute
from sma.runplan import describe
from sma.schemautil import validate_instance
from sma.telemetry import TelemetryRecorder, peak_memory_bytes, read_memory
from tests.fakes import FixtureFetcher, telemetry_factory, write_gguf_snapshot
from tests.harness import make_config, make_services

DATA = Path(__file__).resolve().parent / "data" / "llamacpp"


def _run_dir(tmp_path: Path, run_id: str) -> Path:
    return next((tmp_path / "runs").glob(f"*/*/*/{run_id}"))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# 1. The checkpoint format picks the engine before launch.


def test_the_format_picks_the_engine_and_the_flag_cannot_cross_formats() -> None:
    cfg = default_config()
    assert engine_name(None, cfg, "gguf") == "llamacpp"
    assert engine_name("llamacpp", cfg, "gguf") == "llamacpp"
    assert engine_name(None, cfg, "nvfp4") == "vllm"
    assert engine_name("sglang", cfg, "bf16") == "sglang"
    with pytest.raises(SmaError, match="GGUF files run on llama.cpp"):
        engine_name("vllm", cfg, "gguf")
    with pytest.raises(SmaError, match="llama.cpp runs GGUF files only"):
        engine_name("llamacpp", cfg, "bf16")


def test_preflight_picks_sglang_before_the_run_folder_exists(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, fetcher, _runtime = make_services(tmp_path, cfg)
    fetcher.choose_engine = lambda repo, revision, quant, default: ("sglang", "preflight", "vLLM cannot load Tiny.")
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = _run_dir(tmp_path, run_id)
    manifest = _json(run_dir / "manifest.json")
    assert run_dir.parent.parent.name == "sglang"
    assert (manifest["engine"]["name"], manifest["engine"]["selected_by"]) == ("sglang", "preflight")
    assert manifest["engine"]["selection_reason"] == "vLLM cannot load Tiny."


def test_a_failed_start_is_never_retried_on_another_engine(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg)
    started = []

    def crash(spec):
        started.append(spec)
        raise RuntimeError("engine crashed")

    runtime.start = crash
    with pytest.raises(SmaError, match="vllm failed to become healthy"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    assert started
    for spec in started:
        joined = " ".join(spec.docker_argv)
        assert cfg.engine.image in joined
        assert cfg.engine.sglang_image not in joined and cfg.engine.llamacpp_image not in joined
    manifest = _json(next((tmp_path / "runs").glob("*/*/*/*/manifest.json")))
    assert manifest["engine"]["name"] == "vllm"
    assert manifest["steps"]["serve_quality"]["status"] == "failed"


# 2-4. llama.cpp runs a GGUF repo's quant files, with KL from llama-perplexity.


def _gguf_services(tmp_path: Path, cfg):
    services, _fetcher, runtime = make_services(tmp_path, cfg)
    services.fetcher = FixtureFetcher(write_gguf_snapshot(tmp_path / "gguf"), repo="example/tiny-GGUF", quant_flag="gguf")
    return services, runtime


def test_a_gguf_quant_runs_on_llama_cpp_with_q8_0_kv_and_kl_against_the_bf16_file(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, runtime = _gguf_services(tmp_path, cfg)
    reference = execute(RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="BF16"), cfg, services)
    candidate = execute(
        RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="UD-Q4_K_XL", reference_run_id=reference), cfg, services
    )
    ref_dir, run_dir = _run_dir(tmp_path, reference), _run_dir(tmp_path, candidate)
    assert (run_dir.parent.parent.name, run_dir.parent.name) == ("llamacpp", "gguf-UD-Q4_K_XL")
    assert (ref_dir / "logprobs" / "kld-base.bin").is_file()
    assert not (run_dir / "logprobs" / "kld-base.bin").exists()

    manifest = _json(run_dir / "manifest.json")
    engine = manifest["engine"]
    assert (engine["name"], engine["selected_by"], engine["kv_cache_type"]) == ("llamacpp", "format", "q8_0")
    assert engine["kv_cache_reported"]["quality"]["kv_type"] == "q8_0,q8_0"
    assert engine["tool_call_parser"] == "llama.cpp --jinja"
    assert engine["image"] == cfg.engine.llamacpp_image
    assert manifest["quant_label"] == "UD-Q4_K_XL" and manifest["quant_detected"] == "gguf"
    assert manifest["gguf"]["model_file"] == "Tiny-UD-Q4_K_XL.gguf"
    assert manifest["gguf"]["file_type"] == "MOSTLY_Q4_K_M"
    assert list(manifest["file_hashes"]) == ["Tiny-UD-Q4_K_XL.gguf"]

    quality = next(spec for spec in runtime.specs if spec.name.endswith(f"{candidate}-quality"))
    argv = quality.docker_argv
    assert argv[argv.index("--entrypoint") + 1] == "/app/llama-server"
    assert (argv[argv.index("-ctk") + 1], argv[argv.index("-ctv") + 1]) == ("q8_0", "q8_0")
    assert argv[argv.index("-c") + 1] == str(manifest["memory_plan"]["pool_tokens"])
    assert argv[argv.index("--kv-unified-per-slot") + 1] == str(manifest["memory_plan"]["context"])

    perplexity = [call for call in services.evaluator.calls if "/app/llama-perplexity" in call]
    assert "--kl-divergence" not in perplexity[0] and "--kl-divergence-base" in perplexity[0]
    assert "--kl-divergence" in perplexity[1]
    kl = _json(run_dir / "logprobs" / "kl_vs_reference.json")
    assert (kl["method"], kl["reference_run_id"], kl["kl_mean"]) == ("llama-perplexity-kld", reference, 0.062701)
    assert kl["top1_agreement"] == pytest.approx(0.85973)

    bench = [call for call in services.bencher.calls if "vllm bench serve" in call and candidate in call]
    assert bench and all(call.startswith("docker run --rm --name") and "--network host" in call for call in bench)
    assert all("--tokenizer /model/Tiny-UD-Q4_K_XL.gguf" in call for call in bench)
    assert engine["bench_image"].startswith(cfg.engine.image.split(":")[0])

    summary = _json(run_dir / "summary.json")
    validate_instance(summary, "summary.schema.json")
    validate_instance(manifest, "manifest.schema.json")
    assert (summary["engine"], summary["kl_mean"], summary["perplexity"]) == ("llamacpp", 0.062701, 26.598002)
    assert summary["perplexity_tokens"] == 8184
    assert (run_dir / "determinism.json").is_file()


def test_a_gguf_selector_that_matches_no_file_names_the_ones_that_exist(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _runtime = _gguf_services(tmp_path, cfg)
    with pytest.raises(SmaError, match="BF16"):
        execute(RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="Q3_K_S"), cfg, services)


def test_kl_needs_a_llama_cpp_reference_with_logits(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _runtime = _gguf_services(tmp_path, cfg)
    # A quant file writes no logits, so it cannot be a reference.
    first = execute(RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="UD-Q4_K_XL"), cfg, services)
    second = execute(
        RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="UD-Q4_K_XL", reference_run_id=first), cfg, services
    )
    manifest = _json(_run_dir(tmp_path, second) / "manifest.json")
    assert any("has no logits file" in warning for warning in manifest["warnings"])
    assert not (_run_dir(tmp_path, second) / "logprobs" / "kl_vs_reference.json").exists()


def test_llama_perplexity_logs_parse_into_ppl_and_kl() -> None:
    base = parse_perplexity_log((DATA / "perplexity-base-b11913.log").read_text(encoding="utf-8"))
    assert (base["ppl"], base["chunks"], base["scored_tokens"], base["kld_mean"]) == (25.8602, 8, 8184, None)
    kld = parse_perplexity_log((DATA / "perplexity-kld-b11913.log").read_text(encoding="utf-8"))
    assert (kld["ppl"], kld["ppl_base"], kld["kld_mean"], kld["kld_median"]) == (26.598002, 25.834252, 0.062701, 0.044133)
    assert kld["delta_ppl"] == pytest.approx(0.76375) and kld["same_top_p"] == pytest.approx(0.85973)
    with pytest.raises(SmaError, match="without a perplexity line"):
        parse_perplexity_log("load failed\n")


# 5. The KV cache is always 8-bit.


@pytest.mark.parametrize("flag", ["--kv-cache-dtype", "--max-model-len", "-ctk", "--cache-type-v", "--max-total-tokens", "-c"])
def test_extra_args_cannot_change_the_kv_type_or_the_plan(flag: str) -> None:
    with pytest.raises(SmaError, match=f"SMA sets {flag} itself"):
        validate_config(config_from_mapping({"engine": {"extra_args": [flag, "auto"]}}))
    with pytest.raises(SmaError, match=f"SMA sets {flag} itself"):
        validate_config(config_from_mapping({"engine": {"extra_args": [f"{flag}=f16"]}}))


@pytest.mark.parametrize("quant", ["bf16", "nvfp4"])
def test_vllm_runs_record_the_fp8_cache_and_where_its_scales_come_from(tmp_path: Path, quant: str) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg, quant=quant)
    if quant == "nvfp4":
        path = tmp_path / "snapshot" / "hf_quant_config.json"
        blob = _json(path)
        blob["quantization"]["kv_cache_quant_algo"] = "FP8"
        path.write_text(json.dumps(blob), encoding="utf-8")
    run_id = execute(RunSpec(repo="example/tiny", quant_flag=quant), cfg, services)
    manifest = _json(_run_dir(tmp_path, run_id) / "manifest.json")
    engine = manifest["engine"]
    assert engine["kv_cache_type"] == "fp8"
    assert engine["kv_cache_reported"]["quality"]["kv_type"] == "fp8"
    expected = "checkpoint (FP8)" if quant == "nvfp4" else "engine default (scale 1.0, not calibrated)"
    assert engine["kv_cache_scales"] == expected
    for flags in (engine["flags_quality"], engine["flags_speed"]):
        assert flags[flags.index("--kv-cache-dtype") + 1] == "fp8"


# 6-8. The memory plan: computed before launch, the KV pool sized explicitly, a context per benchmark.


def test_the_plan_sizes_the_server_and_each_suite_records_its_context(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    run_dir = _run_dir(tmp_path, run_id)
    manifest = _json(run_dir / "manifest.json")
    plan = manifest["memory_plan"]
    assert plan["context"] == 40960 and plan["context_reason"].startswith("native maximum")
    assert plan["kv_type"] == "fp8" and plan["budget_gb"] == 100.0
    flags = manifest["engine"]["flags_quality"]
    assert flags[flags.index("--kv-cache-memory-bytes") + 1] == str(plan["pool_bytes"])
    assert flags[flags.index("--max-model-len") + 1] == "40960"
    assert manifest["engine"]["context_per_stream"] == 40960
    scores = _json(run_dir / "quality" / "ifeval" / "scores.json")
    assert scores["context_per_stream"] == plan["benchmarks"]["quality"]
    assert _json(run_dir / "summary.json")["quality"]["ifeval"]["context_per_stream"] == plan["benchmarks"]["quality"]


def test_a_model_that_cannot_fit_the_budget_is_refused_before_any_server_starts(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    cfg.memory.budget_gb = 1.0  # less than the vLLM overhead alone
    services, _fetcher, runtime = make_services(tmp_path, cfg)
    with pytest.raises(SmaError, match="budget"):
        execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    assert runtime.specs == []


def test_the_hub_fetcher_refuses_before_the_weight_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sma.pipeline as pipeline

    cfg = make_config(tmp_path)
    config = _json(Path(__file__).resolve().parent / "data" / "configs" / "qwen3-8b.json")
    fetcher = pipeline.HubFetcher(cfg, registry=lambda _engine, _image: {"Qwen3ForCausalLM"})
    monkeypatch.setattr(fetcher, "_metadata", lambda repo, rev: (None, "a" * 40, ["config.json"], {"model.safetensors": 120 * 10**9}, None))
    monkeypatch.setattr(fetcher, "_config", lambda repo, sha, files, token: (config, None))

    def no_download(*_args, **_kwargs):
        raise AssertionError("the weights were downloaded")

    monkeypatch.setattr(pipeline, "download_snapshot", no_download)
    with pytest.raises(SmaError, match="budget"):
        fetcher.fetch("Qwen/Qwen3-8B", None, "bf16")


# 9. The dry run prints the engine, the KV type, the context and why, and the memory plan.


def test_the_dry_run_prints_the_engine_kv_type_context_and_memory_plan(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    import sma.pipeline as pipeline
    from sma.cli import main
    from sma.memplan import kv_layout_from_hf_config
    from sma.runplan import plan_run

    config = _json(Path(__file__).resolve().parent / "data" / "configs" / "qwen3-30b-a3b-instruct-2507.json")
    cfg = default_config()

    def plan_only(self, repo, revision, quant, gguf):
        if quant == "gguf":
            plan = plan_run(cfg, engine="llamacpp", layout=kv_layout_from_hf_config(config), weights_bytes=17_690_000_000)
            return plan.to_json(), "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"
        return plan_run(cfg, engine="vllm", layout=kv_layout_from_hf_config(config), weights_bytes=61_000_000_000).to_json(), None

    monkeypatch.setattr(pipeline.HubFetcher, "plan_only", plan_only)
    assert main(["run", "Qwen/Qwen3-30B-A3B-Instruct-2507", "--bf16", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "engine: vllm (the default for safetensors" in out
    assert "kv cache: fp8 (always 8-bit)" in out
    assert "memory plan: context per stream: 262,144 tokens (262,144 tokens fits)" in out
    assert "memory plan: memory: 61.0 GB weights + 13.0 GB engine overhead" in out
    assert "--kv-cache-dtype fp8" in out and "--max-model-len 262144" in out

    assert main(["run", "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF", "--gguf", "UD-Q4_K_XL", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "engine: llamacpp (GGUF runs on llama.cpp)" in out
    assert "kv cache: q8_0 (always 8-bit)" in out
    assert "--entrypoint /app/llama-server" in out and "-m /model/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf" in out
    assert "/app/llama-perplexity" in out and "vllm bench serve" in out


# 10. Deployment notes.


def test_the_summary_states_the_deployment_with_computed_and_measured_values(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    def with_memory(run_dir: Path, config) -> TelemetryRecorder:
        recorder = telemetry_factory(run_dir, config)
        recorder.append_memory(110_000_000_000, 130_700_000_000)
        recorder.append_memory(80_000_000_000, 130_700_000_000)
        return recorder

    services.telemetry_factory = with_memory
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    summary = _json(_run_dir(tmp_path, run_id) / "summary.json")
    deployment = summary["deployment"]
    assert (deployment["engine"], deployment["kv_cache_type"], deployment["context_per_stream"]) == ("vllm", "fp8", 40960)
    assert deployment["computed"]["streams_full_context"] >= 1
    assert deployment["measured"]["kv_pool_tokens"] == 1_000_000
    assert deployment["measured"]["streams_full_context"] == 1_000_000 // 40960
    assert deployment["measured"]["peak_memory_gb"] == 30.0
    assert deployment["measured"]["output_tok_s_c4"] == 12.5
    assert deployment["sentence"].startswith("bf16 weights with an FP8 KV cache on vllm:")
    assert "24 full-length streams by the engine's own KV report" in deployment["sentence"]
    assert "Peak memory 30.0 GB." in deployment["sentence"]


def test_peak_memory_is_the_drop_below_the_first_sample(tmp_path: Path) -> None:
    path = tmp_path / "memory.csv"
    path.write_text("ts_unix,mem_available_bytes,mem_total_bytes,phase\n1,100,200,startup\n2,40,200,quality\n3,90,200,speed\n")
    assert peak_memory_bytes(read_memory(path)) == 60
    assert peak_memory_bytes(read_memory(tmp_path / "missing.csv")) is None


# Queue jobs carry the GGUF quant file.


def test_a_queued_gguf_job_keeps_its_quant_file_and_lets_the_format_pick_the_engine(tmp_path: Path) -> None:
    from sma.queue import JobQueue

    queue = JobQueue(tmp_path / "queue")
    job = queue.add(
        [{"repo": "unsloth/X-GGUF", "quant": "gguf", "gguf": "Q8_0", "engine": "llamacpp", "engine_requested": None}],
        kind="single",
    )
    assert (job["legs"][0]["gguf"], job["legs"][0]["engine_requested"]) == ("Q8_0", None)
    with pytest.raises(SmaError, match="needs its quant file"):
        queue.add([{"repo": "unsloth/X-GGUF", "quant": "gguf", "engine": "llamacpp"}], kind="single")


# The logprob dump is streamed: the whole document would not fit next to the KV pool (107 GB peak on a Spark).


def test_the_streaming_json_writer_matches_the_whole_document_and_rejects_nan(tmp_path: Path) -> None:
    from sma.atomic import atomic_write_json_stream

    path = tmp_path / "dump.json"
    seen = []

    def items():
        for index in range(3):
            seen.append(index)
            yield {"index": index, "values": [index, index + 0.5]}

    atomic_write_json_stream(path, {"schema_version": 1}, "windows", items(), lambda: {"count": len(seen)})
    assert _json(path) == {
        "schema_version": 1,
        "windows": [{"index": 0, "values": [0, 0.5]}, {"index": 1, "values": [1, 1.5]}, {"index": 2, "values": [2, 2.5]}],
        "count": 3,
    }
    with pytest.raises(ValueError):
        atomic_write_json_stream(tmp_path / "bad.json", {}, "windows", [float("nan")], dict)
    assert not (tmp_path / "bad.json").exists() and not list(tmp_path.glob(".bad.json.*"))


def test_the_logprob_dump_is_never_built_as_one_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sma.pipeline as pipeline

    whole = pipeline.atomic_write_json

    def refuse_dump(path, payload):
        assert Path(path).name != "dump.json", "the dump went through the whole-document writer"
        whole(path, payload)

    monkeypatch.setattr(pipeline, "atomic_write_json", refuse_dump)
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    dump = _json(_run_dir(tmp_path, run_id) / "logprobs" / "dump.json")
    assert len(dump["windows"]) == cfg.logprobs.num_windows and dump["token_count"] > 0


def test_a_split_gguf_set_is_one_quant_hashed_and_loaded_from_its_first_file(tmp_path: Path) -> None:
    from sma.fetch import inspect_gguf
    from tests.gguf_writer import write_gguf

    root = tmp_path / "repo" / "UD-Q4_K_XL"
    root.mkdir(parents=True)
    meta = {
        "general.architecture": "qwen3",
        "general.file_type": 15,
        "split.count": 2,
        "qwen3.block_count": 2,
        "qwen3.context_length": 40960,
        "qwen3.embedding_length": 32,
        "qwen3.attention.head_count": 4,
        "qwen3.attention.head_count_kv": 2,
        "tokenizer.ggml.tokens": ["a", "b"],
    }
    for part in (1, 2):
        write_gguf(root / f"Big-UD-Q4_K_XL-0000{part}-of-00002.gguf", {**meta, "split.no": part - 1},
                   [(f"blk.{part}.weight", (256, 4), 12)], data=True)
    (tmp_path / "repo" / "Big-Q8_0.gguf").write_bytes(b"not read")
    result = inspect_gguf(tmp_path / "repo", repo="org/Big-GGUF", revision_sha="a" * 40, selector="UD-Q4_K_XL")
    assert result.gguf["files"] == ["UD-Q4_K_XL/Big-UD-Q4_K_XL-00001-of-00002.gguf", "UD-Q4_K_XL/Big-UD-Q4_K_XL-00002-of-00002.gguf"]
    assert result.gguf["model_file"].endswith("-00001-of-00002.gguf") and result.gguf["split_count"] == 2
    assert sorted(result.file_hashes) == result.gguf["files"]
    assert result.params_total == 2 * 256 * 4
    assert result.weights_bytes == sum(result.file_sizes.values())


def test_bfcl_runs_on_llama_cpp_with_the_chat_templates_tool_calls(tmp_path: Path) -> None:
    from sma.config import QualitySuite

    cfg = make_config(tmp_path)
    cfg.quality.suites = [QualitySuite("bfcl_v3", image="nvcr.io/nvidia/eval-factory/bfcl:26.03", task="bfclv3_ast", tools=True)]
    services, _runtime = _gguf_services(tmp_path, cfg)
    run_id = execute(RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="UD-Q4_K_XL"), cfg, services)
    scores = _json(_run_dir(tmp_path, run_id) / "quality" / "bfcl_v3" / "scores.json")
    assert scores["primary_score"] is not None and scores["context_per_stream"] == 40960


def test_a_gguf_run_resumes_with_its_quant_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sma.pipeline as pipeline
    from sma.cli import main

    cfg = make_config(tmp_path)
    services, _runtime = _gguf_services(tmp_path, cfg)
    services.hooks.kill_step = "quality"
    with pytest.raises(SimulatedKill, match="killed mid-step"):
        execute(RunSpec(repo="example/tiny-GGUF", quant_flag="gguf", gguf="UD-Q4_K_XL"), cfg, services)
    run_id = next((tmp_path / "runs").glob("*/llamacpp/gguf-UD-Q4_K_XL/*")).name
    seen = {}

    def fake_execute(spec, config, _services, **_kwargs):
        seen.update(gguf=spec.gguf, engine=spec.engine, quant=spec.quant_flag)
        return spec.resume_run_id

    monkeypatch.setattr(pipeline, "execute", fake_execute)
    monkeypatch.setattr(pipeline, "production_services", lambda _cfg: services)
    config_path = tmp_path / "sma.yaml"
    config_path.write_text(f"runs_dir: {tmp_path / 'runs'}\nhost_alias: {cfg.host_alias}\n", encoding="utf-8")
    assert main(["resume", run_id, "--config", str(config_path)]) == 0
    assert seen == {"gguf": "UD-Q4_K_XL", "engine": "llamacpp", "quant": "gguf"}


def test_llama_cpp_plans_its_buffers_and_its_prompt_cache() -> None:
    # llama-server's compute buffers grow with its whole KV cache, about 6 KB per token for Qwen3 on a
    # Spark, and its host-RAM prompt cache (8 GiB by default) grows once prompts arrive. Unplanned, a
    # run peaked at 110.6 GB.
    from sma.memplan import kv_layout_from_hf_config
    from sma.runplan import llamacpp_buffer_bytes_per_token, plan_run

    cfg = default_config()
    layout = kv_layout_from_hf_config(_json(Path(__file__).resolve().parent / "data" / "configs" / "qwen3-8b.json"))
    llama = plan_run(cfg, engine="llamacpp", layout=layout, weights_bytes=1_200_000_000)
    vllm = plan_run(cfg, engine="vllm", layout=layout, weights_bytes=1_200_000_000)
    assert llama.overhead_bytes == 3_000_000_000 + 8192 * 2**20
    # Qwen3's 8 KV heads of 128 measured 6,164 bytes per token (Qwen3-0.6B, same per-layer K and V).
    assert llamacpp_buffer_bytes_per_token(layout) == llama.buffer_bytes_per_token == 6144
    assert llama.buffer_bytes == llama.pool_tokens * 6144
    assert llama.footprint_bytes == 1_200_000_000 + llama.overhead_bytes + llama.pool_bytes + llama.buffer_bytes
    assert (vllm.buffer_bytes, vllm.pool_streams) == (0, None)
    assert vllm.footprint_bytes == 100e9
    argv = _serve_llama(cfg)
    assert argv[argv.index("--cache-ram") + 1] == "8192"


# LFM3-1.7B-GGUF's layout, as SMA reads it from the GGUF metadata.
_LFM3_1_7B = KvLayout(
    total_layers=24,
    full_attention_layers=6,
    full_values_per_token=6144,
    sliding_attention_layers=5,
    sliding_values_per_token=5120,
    sliding_window=256,
    state_bytes_per_stream=638_976,
    native_context=262_144,
    native_source="lfm3.context_length",
    rope_stretched=False,
)


def test_llama_cpp_pool_holds_its_slots_not_the_rest_of_the_budget() -> None:
    # On a Spark (2026-10-28), LFM3-1.7B BF16 got the rest of the 100 GB budget as its pool: -c 11,618,008
    # for 8 slots of 262,144, 5.5 times what they can use. llama.cpp sized its compute buffers by that
    # cache, 4,100 bytes per token, 47.6 GB the plan did not count. The run peaked at 127 GB of 130.7 GB,
    # and the next run's server took the machine down.
    from sma.memplan import stream_kv_bytes
    from sma.runplan import llamacpp_buffer_bytes_per_token, plan_run, slots

    cfg = default_config()
    assert llamacpp_buffer_bytes_per_token(_LFM3_1_7B) == 4096
    plan = plan_run(cfg, engine="llamacpp", layout=_LFM3_1_7B, weights_bytes=3_470_000_000)
    assert (plan.kv_bytes_per_token, plan.context) == (6528, 262_144)
    assert plan.pool_streams == slots(cfg) == 8
    assert plan.pool_bytes == 8 * stream_kv_bytes(_LFM3_1_7B, 262_144, "q8_0")
    assert 8 * 262_144 <= plan.pool_tokens < 8 * 262_144 * 1.01
    assert plan.streams_full_context == 8
    assert plan.buffer_bytes == plan.pool_tokens * 4096
    assert plan.footprint_bytes == 3_470_000_000 + plan.overhead_bytes + plan.pool_bytes + plan.buffer_bytes
    assert plan.footprint_bytes < 40e9
    # Determinism's 64 prompts share the 8 slots' pool at a shorter step (each stream also keeps its
    # convolution state, so 64 at 32,768 just miss).
    assert plan.benchmarks["determinism"] == 16_384
    assert any(line.startswith("KV pool: 8 slots at 262,144 tokens") for line in describe(plan.to_json()))


def test_llama_cpp_pool_is_the_budget_when_its_slots_need_more() -> None:
    # Qwen3-30B-A3B-2507 at 262,144: 8 slots would need 109 GB of Q8_0 cache, so the budget sets the pool.
    from sma.memplan import kv_layout_from_hf_config
    from sma.runplan import plan_run

    layout = kv_layout_from_hf_config(_json(Path(__file__).resolve().parent / "data" / "configs" / "qwen3-30b-a3b-instruct-2507.json"))
    plan = plan_run(default_config(), engine="llamacpp", layout=layout, weights_bytes=17_000_000_000)
    assert plan.pool_streams is None
    assert plan.pool_bytes < 8 * plan.stream_bytes
    assert plan.buffer_bytes == plan.pool_tokens * plan.buffer_bytes_per_token > 0
    assert 99e9 < plan.footprint_bytes <= 100e9


def _serve_llama(cfg) -> list[str]:
    from sma.commands import serve_argv
    from tests.harness import SERVE_PLAN

    return serve_argv(cfg, mode="speed", served_name="m", is_moe=False, quant_flag="gguf", moe_backend=None,
                      batch_invariant=False, plan=SERVE_PLAN, engine="llamacpp")


def test_declared_kv_scales_that_the_engine_used_as_1_0_are_recorded_as_such(tmp_path: Path) -> None:
    # nvidia/Qwen3-8B-NVFP4 declares FP8 KV scales, and every one is 1.0; vLLM logs that it used 1.0.
    cfg = make_config(tmp_path)
    services, _fetcher, runtime = make_services(tmp_path, cfg, quant="nvfp4")
    path = tmp_path / "snapshot" / "hf_quant_config.json"
    blob = _json(path)
    blob["quantization"]["kv_cache_quant_algo"] = "FP8"
    path.write_text(json.dumps(blob), encoding="utf-8")
    runtime.extra_log = "WARNING [kv_cache.py:123] Using KV cache scaling factor 1.0 for fp8_e4m3.\n"
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="nvfp4"), cfg, services)
    engine = _json(_run_dir(tmp_path, run_id) / "manifest.json")["engine"]
    assert engine["kv_cache_scales"] == "checkpoint (FP8); the engine log says it used a KV scale of 1.0"
    assert engine["kv_cache_reported"]["quality"]["scale_1_0"] is True


def test_a_run_whose_measured_peak_passed_the_budget_says_so(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)

    def over(run_dir: Path, config) -> TelemetryRecorder:
        recorder = telemetry_factory(run_dir, config)
        recorder.append_memory(120_000_000_000, 130_700_000_000)
        recorder.append_memory(18_000_000_000, 130_700_000_000)
        return recorder

    services.telemetry_factory = over
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    warnings = _json(_run_dir(tmp_path, run_id) / "summary.json")["warnings"]
    assert any("Peak memory was 102.0 GB, over the 100 GB budget" in w and "memory.overhead_gb.vllm" in w for w in warnings)


@pytest.mark.parametrize("quant", ["nvfp4", "mxfp4"])
def test_quantized_checkpoints_run_their_real_kernels_on_vllm(tmp_path: Path, quant: str) -> None:
    # vLLM's batch-invariant mode served NVFP4 by emulation on a Spark, at about 2 tokens per second.
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg, quant=quant)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag=quant), cfg, services)
    manifest = _json(_run_dir(tmp_path, run_id) / "manifest.json")
    engine = manifest["engine"]
    assert engine["batch_invariant_quality"] is False
    assert "VLLM_BATCH_INVARIANT" not in engine["env_quality"]
    assert "--no-async-scheduling" not in engine["flags_quality"]
    assert engine["batch_invariant_warning"].startswith(f"Batch invariance is off for this {quant} checkpoint")
    assert (_run_dir(tmp_path, run_id) / "determinism.json").is_file()


def test_bf16_on_vllm_keeps_batch_invariance(tmp_path: Path) -> None:
    cfg = make_config(tmp_path)
    services, _fetcher, _runtime = make_services(tmp_path, cfg)
    run_id = execute(RunSpec(repo="example/tiny", quant_flag="bf16"), cfg, services)
    engine = _json(_run_dir(tmp_path, run_id) / "manifest.json")["engine"]
    assert engine["batch_invariant_quality"] is True and engine["env_quality"]["VLLM_BATCH_INVARIANT"] == "1"


def test_a_reasoning_parser_reaches_vllm_and_sglang_but_not_llama_cpp() -> None:
    # LFM3 on SGLang thinks by default; --reasoning-parser qwen3 moves the reasoning out of the answer the
    # agent steps read. llama.cpp does the same with --jinja.
    from sma.commands import serve_argv
    from tests.harness import SERVE_PLAN

    cfg = default_config()
    cfg.engine.reasoning_parser = "qwen3"
    for engine, quant in (("vllm", "bf16"), ("sglang", "bf16"), ("llamacpp", "gguf")):
        argv = serve_argv(cfg, mode="quality", served_name="m", is_moe=False, quant_flag=quant, moe_backend=None,
                          batch_invariant=False, plan=SERVE_PLAN, engine=engine)
        if engine == "llamacpp":
            assert "--reasoning-parser" not in argv
        else:
            assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    cfg.engine.reasoning_parser = None
    argv = serve_argv(cfg, mode="quality", served_name="m", is_moe=False, quant_flag="bf16", moe_backend=None,
                      batch_invariant=False, plan=SERVE_PLAN, engine="sglang")
    assert "--reasoning-parser" not in argv


def test_sglang_plans_sliding_layers_in_full_and_its_memory_share_holds_the_plan() -> None:
    # On a Spark (2026-10-30) SGLang dev-lfm3 gave LFM3-1.7B's 5 sliding-window layers full-length K and V:
    # 9.39M tokens took 98.5 GB, the plan said 83.5 GB, and SMA's memory guard killed it at 9 GB free.
    from sma.backends import ServePlan, sglang_mem_fraction
    from sma.config import default_config
    from sma.memplan import kv_layout_from_hf_config
    from sma.runplan import plan_run

    cfg = default_config()
    layout = kv_layout_from_hf_config(_json(Path(__file__).resolve().parent / "data" / "configs" / "lfm3-1.7b.json"))
    assert (layout.full_attention_layers, layout.sliding_attention_layers, layout.sliding_window) == (6, 5, 256)
    vllm = plan_run(cfg, engine="vllm", layout=layout, weights_bytes=3_460_000_000)
    sglang = plan_run(cfg, engine="sglang", layout=layout, weights_bytes=3_460_000_000)
    # 4 KV heads x 128 x K and V x 1 byte (FP8) = 1,024 bytes a layer: 6 layers for vLLM's plan, 11 for SGLang's.
    assert (vllm.kv_bytes_per_token, sglang.kv_bytes_per_token) == (6_144, 11_264)
    assert sglang.pool_tokens == sglang.pool_bytes // 11_264

    plan = ServePlan(context=sglang.context, pool_bytes=sglang.pool_bytes, pool_tokens=sglang.pool_tokens, weights_bytes=3_460_000_000)
    # 3.5 GB in use already: SGLang may hold that, the weights, and the pool, and no more.
    fraction = sglang_mem_fraction(cfg, plan, memory=(127_160_000_000, 130_660_000_000))
    assert fraction == pytest.approx((3_500_000_000 + 3_460_000_000 + sglang.pool_bytes) / 130_660_000_000)
    assert fraction < cfg.engine.gpu_memory_utilization
    # Never above engine.gpu_memory_utilization, and that cap before a plan exists.
    assert sglang_mem_fraction(cfg, plan, memory=(20_000_000_000, 130_660_000_000)) == cfg.engine.gpu_memory_utilization
    assert sglang_mem_fraction(cfg, ServePlan("<context>", "<kv-pool-bytes>", "<kv-pool-tokens>")) == cfg.engine.gpu_memory_utilization
