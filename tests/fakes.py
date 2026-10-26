"""In-process stand-ins for Docker, Hugging Face, lm-eval, and vLLM bench."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from sma.fetch import FetchResult, inspect_snapshot
from sma.runtime import StartSpec
from sma.telemetry import TelemetryRecorder
from tests.fake_vllm import FakeVLLM


def write_safetensors(path: Path, tensors: dict[str, list[int]]) -> None:
    header: dict[str, Any] = {
        name: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
        for name, shape in tensors.items()
    }
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header).encode()
    pad = (8 - (len(blob) % 8)) % 8
    blob += b" " * pad
    path.write_bytes(struct.pack("<Q", len(blob)) + blob)


def write_snapshot(
    root: Path,
    *,
    quant: str = "bf16",
    moe: bool = False,
    model_type: str = "llama",
    architectures: list[str] | None = None,
    tensors: dict[str, list[int]] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config: dict[str, Any] = {
        "model_type": model_type,
        "architectures": architectures or ["LlamaForCausalLM"],
        "torch_dtype": "bfloat16",
    }
    if moe:
        config["num_experts"] = 4
        config["num_experts_per_tok"] = 2
        config["model_type"] = "qwen2_moe"
    if quant == "nvfp4":
        (root / "hf_quant_config.json").write_text(
            json.dumps(
                {
                    "quantization": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                        "kv_cache_quant_algo": None,
                        "exclude_modules": ["lm_head"],
                    },
                    "producer": {"name": "example-recipe", "version": "0"},
                }
            ),
            encoding="utf-8",
        )
    elif quant == "mxfp4":
        config["quantization_config"] = {"quant_method": "mxfp4"}
    elif quant == "compressed-nvfp4":
        config["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                    "input_activations": {"num_bits": 4, "type": "float", "group_size": 16},
                }
            },
        }
    elif quant == "dgpp_mixed":
        config["quantization_config"] = {
            "quant_method": "dgpp_mixed",
            "activations": "bfloat16",
            "routed_experts": {
                "precision_by_layer": {
                    "2": {"gate": "fp8", "up": "fp8", "down": "fp8"},
                    "12": {"gate": "fp8", "up": "nvfp4", "down": "fp8"},
                    "31": {"gate": "nvfp4", "up": "nvfp4", "down": "nvfp4"},
                }
            },
        }
    elif quant == "w4a16":
        (root / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {"quant_algo": "W4A16_NVFP4"}}),
            encoding="utf-8",
        )
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "tokenizer.json").write_bytes(b'{"version":"1.0","truncation":null,"padding":null}')
    shapes = tensors or {"model.embed.weight": [32, 16]}
    if moe and tensors is None:
        shapes = {
            "model.embed.weight": [32, 16],
            "model.layers.0.mlp.experts.0.gate.weight": [16, 16],
            "model.layers.0.mlp.experts.1.gate.weight": [16, 16],
            "model.layers.0.mlp.experts.2.gate.weight": [16, 16],
            "model.layers.0.mlp.experts.3.gate.weight": [16, 16],
        }
    write_safetensors(root / "model.safetensors", shapes)
    return root


class FixtureFetcher:
    def __init__(self, snapshot: Path, *, repo: str, quant_flag: str, sha: str = "a" * 40) -> None:
        self.snapshot = snapshot
        self.repo = repo
        self.quant_flag = quant_flag
        self.sha = sha
        self.calls = 0

    def fetch(self, repo: str, revision: str | None, quant_flag: str) -> FetchResult:
        self.calls += 1
        return inspect_snapshot(
            self.snapshot,
            repo=repo,
            revision_sha=self.sha,
            quant_flag=quant_flag,
            revision_requested=revision,
        )


class FakeRuntime:
    def __init__(self) -> None:
        self.fail_backends: set[str] = set()
        self.specs: list[StartSpec] = []
        self.running: set[str] = set()
        self.stop_count = 0
        self.digest = "sha256:" + "ab" * 32
        self.server: FakeVLLM | None = None

    def ensure_image(self, image: str | None = None) -> str:
        return self.digest

    def start(self, spec: StartSpec) -> None:
        self.specs.append(spec)
        backend = _backend(spec)
        if backend in self.fail_backends:
            raise RuntimeError(f"{backend} crashed")
        if self.server is not None:
            self.server.stop()
        self.server = FakeVLLM(spec.port)
        self.server.nondeterministic = False
        self.server.start()
        self.running.add(spec.name)

    def stop(self, name: str) -> None:
        self.stop_count += 1
        self.running.discard(name)
        if self.server is not None and not self.running:
            self.server.stop()
            self.server = None

    def logs(self, name: str) -> str:
        return "Using moe backend from the explicit flag\n"

    def is_running(self, name: str) -> bool:
        return name in self.running

    def list_sma(self) -> list[str]:
        return sorted(self.running)


def _backend(spec: StartSpec) -> str | None:
    argv = spec.serve_argv
    if "--moe-backend" not in argv:
        return None
    return argv[argv.index("--moe-backend") + 1]


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, argv: list[str], output_dir: Path, timeout_s: float, log_path: Path) -> None:
        self.calls.append(list(argv))
        output_dir.mkdir(parents=True, exist_ok=True)
        suite = output_dir.name
        (output_dir / "scores.json").write_text(
            json.dumps(
                {
                    "scorer": "lm-eval-harness",
                    "suite": suite,
                    "primary_metric": "acc,none",
                    "primary_score": 0.5,
                    "limit": 2,
                    "results": {suite: {"acc,none": 0.5}},
                }
            ),
            encoding="utf-8",
        )
        (output_dir / "samples.jsonl").write_text('{"doc_id": 0, "exact": true}\n', encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake lm-eval\n", encoding="utf-8")


class FakeBencher:
    def __init__(self, *, ttft_ms: float = 100.0, tpot_ms: float = 20.0, output_tok_s: float = 12.5) -> None:
        self.calls: list[str] = []
        self.ttft_ms = ttft_ms
        self.tpot_ms = tpot_ms
        self.output_tok_s = output_tok_s

    def run(self, argv: list[str], result_path: Path, timeout_s: float, log_path: Path) -> None:
        self.calls.append(" ".join(argv))
        target = bench_output_host_path(argv, result_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "p50_ttft_ms": self.ttft_ms,
                    "p90_ttft_ms": self.ttft_ms + 10,
                    "p99_ttft_ms": self.ttft_ms + 20,
                    "p50_tpot_ms": self.tpot_ms,
                    "p90_tpot_ms": self.tpot_ms + 2,
                    "p99_tpot_ms": self.tpot_ms + 4,
                    "p50_itl_ms": self.tpot_ms,
                    "p90_itl_ms": self.tpot_ms + 1,
                    "p99_itl_ms": self.tpot_ms + 3,
                    "p50_e2el_ms": 400.0,
                    "p90_e2el_ms": 500.0,
                    "p99_e2el_ms": 600.0,
                    "output_throughput": self.output_tok_s,
                }
            ),
            encoding="utf-8",
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fake bench\n", encoding="utf-8")


def container_bench_output(argv: list[str]) -> str | None:
    """The output file named by a vLLM or SGLang bench command."""
    if "--output-file" in argv:
        return argv[argv.index("--output-file") + 1]
    if "--result-dir" in argv and "--result-filename" in argv:
        directory = argv[argv.index("--result-dir") + 1]
        name = argv[argv.index("--result-filename") + 1]
        return str(Path(directory) / name)
    return None


def bench_output_host_path(argv: list[str], result_path: Path) -> Path:
    """Map /sma-run/... from the command onto the run directory SMA will read.

    The file is written only where the command says. If that is not the attempt
    directory SMA reads, the bench JSON stays missing and the cell cannot
    invent a tok/s number.
    """
    declared = container_bench_output(argv)
    if not declared:
        raise RuntimeError("bench command does not name an output file")
    normalized = declared.replace("\\", "/")
    marker = "/sma-run/"
    if marker not in normalized:
        return Path(declared)
    relative = normalized.split(marker, 1)[1]
    # result_path is <run>/speed/cells/<cell>/attempt-N/bench.json
    if len(result_path.parents) < 5:
        return Path(declared)
    return result_path.parents[4] / relative


def telemetry_factory(run_dir: Path, cfg: Any) -> TelemetryRecorder:
    recorder = TelemetryRecorder(run_dir / "telemetry.csv", cfg)
    recorder.start(sampler=False)
    original = recorder.set_phase

    def set_phase(phase: str) -> None:
        original(phase)
        if phase.startswith("speed:"):
            recorder.append_parsed(
                "1500, 45, 80, Not Active, Not Active, Not Active, Not Active",
                phase=phase,
            )

    recorder.set_phase = set_phase  # type: ignore[method-assign]
    recorder.append_parsed("1500, 40, 80, Not Active, Not Active, Not Active, Not Active", phase="startup")
    return recorder
