"""Serving rates from sampled engine counters: the recorder's rows, the window math, each engine's rates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sma.backends import get_backend
from sma.config import default_config
from sma.serving import Series, recent_series, window
from sma.telemetry import ENGINE_METRICS_FILE, TelemetryRecorder

LLAMA = "llamacpp:"


def _llama(prompt: float, prompt_s: float, predicted: float, predicted_s: float, busy: float) -> dict[str, float]:
    return {
        LLAMA + "prompt_tokens_total": prompt,
        LLAMA + "prompt_seconds_total": prompt_s,
        LLAMA + "tokens_predicted_total": predicted,
        LLAMA + "tokens_predicted_seconds_total": predicted_s,
        LLAMA + "requests_processing": busy,
    }


SERIES = Series(
    "llamacpp",
    [
        (1000.0, _llama(1000, 1.0, 10_000, 300, 8)),
        (1015.0, _llama(3000, 2.0, 14_000, 400, 8)),
        # The server restarted: counters start again from zero.
        (1030.0, _llama(500, 0.25, 1000, 25, 4)),
    ],
)


def test_llama_cpp_rates_over_a_window_count_a_restart_from_zero() -> None:
    (found,) = window([SERIES], start=0, max_gap_s=45)
    rates = found.rates()
    # Prompt +2000 then +500 tokens in 1.0 + 0.25 s; generated +4000 then +1000 in 100 + 25 s.
    assert rates["prefill_tok_s"] == pytest.approx(2000.0)
    assert rates["decode_tok_s"] == pytest.approx(40.0)
    assert rates["itl_ms"] == pytest.approx(25.0)
    assert rates["ttft_ms"] is None  # llama-server does not report it.
    assert rates["generated_tok_s_all"] == pytest.approx(5000 / 30)
    assert rates["prompt_tok_s_all"] == pytest.approx(2500 / 30)
    assert rates["in_flight"] == pytest.approx(6.0)
    assert found.seconds == 30


def test_window_counts_only_pairs_that_end_inside_it_and_skips_gaps() -> None:
    (recent,) = window([SERIES], start=1020, max_gap_s=45)
    assert recent.seconds == 15 and recent.change[LLAMA + "tokens_predicted_total"] == 1000
    assert window([SERIES], start=1020, max_gap_s=10) == []
    assert window([SERIES], start=2000, max_gap_s=45) == []


def test_vllm_and_sglang_report_ttft_and_itl_from_their_histograms() -> None:
    change = {
        "vllm:time_to_first_token_seconds_sum": 3.0,
        "vllm:time_to_first_token_seconds_count": 10,
        "vllm:inter_token_latency_seconds_sum": 2.0,
        "vllm:inter_token_latency_seconds_count": 100,
        "vllm:prompt_tokens_total": 5000,
        "vllm:request_prefill_time_seconds_sum": 2.5,
        "vllm:generation_tokens_total": 1000,
        "vllm:request_decode_time_seconds_sum": 20,
    }
    rates = get_backend("vllm").serving_rates(change, {"vllm:num_requests_running": 3}, 10)
    assert (rates["ttft_ms"], rates["itl_ms"], rates["prefill_tok_s"], rates["decode_tok_s"]) == (300, 20, 2000, 50)
    assert (rates["generated_tok_s_all"], rates["in_flight"]) == (100, 3)
    sglang = get_backend("sglang").serving_rates(
        {"sglang:inter_token_latency_seconds_sum": 1.0, "sglang:inter_token_latency_seconds_count": 50}, {}, 10
    )
    assert (sglang["itl_ms"], sglang["decode_tok_s"], sglang["prefill_tok_s"], sglang["ttft_ms"]) == (20, 50, None, None)


def test_recorder_writes_only_the_engines_counters_summed_over_labels(tmp_path: Path) -> None:
    recorder = TelemetryRecorder(tmp_path / "telemetry.csv", default_config())
    assert recorder.sample_engine(text="llamacpp:prompt_tokens_total 1") is False  # No engine yet.
    recorder.watch_engine("http://127.0.0.1:9", "/metrics", get_backend("llamacpp").serving_metrics())
    recorder.set_phase("quality")
    scrape = (
        "# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed\n"
        'llamacpp:prompt_tokens_total{slot="0"} 10\n'
        'llamacpp:prompt_tokens_total{slot="1"} 5\n'
        "llamacpp:requests_processing 2\n"
        "process_cpu_seconds_total 99\n"
    )
    assert recorder.sample_engine(now=1793239534.5, text=scrape) is True
    assert recorder.sample_engine(text="process_cpu_seconds_total 1\n") is False
    (row,) = [json.loads(line) for line in (tmp_path / ENGINE_METRICS_FILE).read_text().splitlines()]
    assert row == {
        "ts": 1793239534.5,
        "iso": "2026-10-29T02:05:34Z",
        "phase": "quality",
        "values": {"llamacpp:prompt_tokens_total": 15.0, "llamacpp:requests_processing": 2.0},
    }
    # No server answers: no row, no error.
    assert recorder.sample_engine() is False


def test_recent_series_reads_each_runs_samples_with_its_engine(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "Org_Model-GGUF" / "llamacpp" / "gguf-BF16" / "20261029T020534Z-aaaa"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"engine": {"name": "llamacpp"}}))
    rows = [{"ts": t, "values": v} for t, v in SERIES.samples]
    (run_dir / ENGINE_METRICS_FILE).write_text("\n".join(json.dumps(r) for r in rows) + '\n{"ts": 1031, "val')
    (series,) = recent_series(tmp_path / "runs", now=1035, span_s=86400)
    assert series.engine == "llamacpp" and len(series.samples) == 3  # The cut-short last row is skipped.
    assert recent_series(tmp_path / "runs", now=1035, span_s=20)[0].samples == SERIES.samples[1:]


def test_rates_over_two_engines_are_one_average_from_summed_tokens_and_seconds() -> None:
    # Owner, 2026-10-29: sma watch shows one SERVING table, whatever engine served.
    from sma.serving import Series, combined_rates, window

    llamacpp = Series("llamacpp", [
        (0.0, {"llamacpp:tokens_predicted_total": 0, "llamacpp:tokens_predicted_seconds_total": 0}),
        (10.0, {"llamacpp:tokens_predicted_total": 100, "llamacpp:tokens_predicted_seconds_total": 10}),
    ])  # fmt: skip
    sglang = Series("sglang", [
        (20.0, {"sglang:inter_token_latency_seconds_count": 0, "sglang:inter_token_latency_seconds_sum": 0, "sglang:generation_tokens_total": 0}),
        (30.0, {"sglang:inter_token_latency_seconds_count": 300, "sglang:inter_token_latency_seconds_sum": 10, "sglang:generation_tokens_total": 300}),
    ])  # fmt: skip
    rates = combined_rates(window([llamacpp, sglang], start=0.0, max_gap_s=60.0))
    # 400 tokens in 20 decode-seconds: 20 tok/s per stream, 50 ms between tokens; 400 tokens in 20 s of wall time.
    assert rates["decode_tok_s"] == 20.0 and rates["itl_ms"] == 50.0 and rates["generated_tok_s_all"] == 20.0
    # Neither reported prefill here; SGLang never does. Not measured, not zero.
    assert rates["prefill_tok_s"] is None


def test_sglang_prefill_comes_from_its_per_stage_latency_and_only_the_prefill_stage_counts() -> None:
    # SGLang dev-lfm3 on a Spark (2026-10-30) reports each request's prefill time as one stage of a labelled
    # metric; v0.5.10 had none. Summing every stage would count request processing as prefill.
    from sma.telemetry import engine_values

    text = "\n".join([
        'sglang:prompt_tokens_total{is_streaming="false",model_name="m"} 171986.0',
        'sglang:per_stage_req_latency_seconds_sum{model_name="m",stage="request_process"} 0.0153',
        'sglang:per_stage_req_latency_seconds_sum{model_name="m",stage="prefill_forward"} 61.4436',
        'sglang:inter_token_latency_seconds_sum{model_name="m"} 10.0',
        'sglang:inter_token_latency_seconds_count{model_name="m"} 400.0',
    ])  # fmt: skip
    backend = get_backend("sglang")
    values = engine_values(text, backend.serving_metrics())
    assert values['sglang:per_stage_req_latency_seconds_sum{stage="prefill_forward"}'] == 61.4436
    assert backend.serving_rates(values, {}, 100.0)["prefill_tok_s"] == pytest.approx(171986 / 61.4436)
    # An SGLang without the stage reports no prefill rate: not measured, not zero.
    older = engine_values("\n".join(line for line in text.splitlines() if "per_stage" not in line), backend.serving_metrics())
    assert backend.serving_rates(older, {}, 100.0)["prefill_tok_s"] is None
