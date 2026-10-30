"""In-process stand-ins for Docker, Hugging Face, NeMo Evaluator, and vLLM bench."""

from __future__ import annotations

import json
import socket
import struct
from pathlib import Path
from typing import Any

import yaml

from sma.errors import SmaError
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
        # What the memory plan reads: a tiny model with Qwen3's 40,960-token native context.
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_size": 32,
        "max_position_embeddings": 40960,
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


def write_gguf_snapshot(root: Path, selectors: tuple[str, ...] = ("BF16", "UD-Q4_K_XL")) -> Path:
    """An Unsloth-style GGUF repo: one small qwen3 file per quant selector, with tokenizer metadata."""
    from tests.gguf_writer import write_gguf

    root.mkdir(parents=True, exist_ok=True)
    file_types = {"BF16": 32, "F16": 1, "Q8_0": 7}
    for selector in selectors:
        meta = {
            "general.architecture": "qwen3",
            "general.name": "Tiny",
            "general.file_type": file_types.get(selector, 15),
            "qwen3.block_count": 2,
            "qwen3.context_length": 40960,
            "qwen3.embedding_length": 32,
            "qwen3.attention.head_count": 4,
            "qwen3.attention.head_count_kv": 2,
            "qwen3.attention.key_length": 8,
            "qwen3.attention.value_length": 8,
            "tokenizer.ggml.model": "gpt2",
            "tokenizer.ggml.tokens": ["a", "b", "c"],
        }
        tensor_type = 30 if selector == "BF16" else 12  # BF16, or Q4_K
        write_gguf(root / f"Tiny-{selector}.gguf", meta, [("token_embd.weight", (256, 4), tensor_type)], data=True)
    (root / "README.md").write_text("tiny GGUF repo\n", encoding="utf-8")
    return root


class FixtureFetcher:
    def __init__(self, snapshot: Path, *, repo: str, quant_flag: str, sha: str = "a" * 40) -> None:
        self.snapshot = snapshot
        self.repo = repo
        self.quant_flag = quant_flag
        self.sha = sha
        self.calls = 0
        self.tokenizers: dict[str, Path] = {}
        self.tokenizer_requests: list[tuple[str, str | None]] = []

    def tokenizer_snapshot(self, repo: str, revision: str | None = None) -> tuple[Path, str]:
        """A repo's tokenizer files: from tokenizers[repo], a folder a test fills."""
        self.tokenizer_requests.append((repo, revision))
        if repo not in self.tokenizers:
            raise SmaError(f"no tokenizer fixture for {repo}")
        return self.tokenizers[repo], "b" * 40

    def fetch(self, repo: str, revision: str | None, quant_flag: str, gguf: str | None = None) -> FetchResult:
        self.calls += 1
        if gguf is not None:
            from sma.fetch import inspect_gguf

            return inspect_gguf(self.snapshot, repo=repo, revision_sha=self.sha, selector=gguf, revision_requested=revision)
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
        self.gguf_archs: set[str] | None = None
        self.specs: list[StartSpec] = []
        self.running: set[str] = set()
        self.stop_count = 0
        self.followed: list[str] = []
        self.exited: set[str] = set()
        self.digest = "sha256:" + "ab" * 32
        self.server: FakeVLLM | None = None
        # Appended to every container log, for a test that needs one more engine line.
        self.extra_log = ""

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

    def kill(self, name: str) -> None:
        """The server stops on its own, as when the host stops it. The container still exists."""
        self.running.discard(name)
        self.exited.add(name)
        if self.server is not None:
            self.server.stop()
            self.server = None

    def stop(self, name: str) -> None:
        self.stop_count += 1
        self.running.discard(name)
        self.exited.discard(name)
        if self.server is not None and not self.running:
            self.server.stop()
            self.server = None

    def logs(self, name: str) -> str:
        # Each engine's own KV lines; each backend reads only its own.
        return (
            "Using moe backend from the explicit flag\n"
            "INFO [cache.py:280] Using fp8 data type to store kv cache. It reduces the GPU memory footprint\n"
            "INFO [kv_cache_utils.py:1731] GPU KV cache size: 1,000,000 tokens\n"
            "[sglang] Using KV cache dtype: torch.float8_e4m3fn\n"
            "[sglang] KV Cache is allocated. #tokens: 1000000, K size: 1.00 GB\n"
            "I llama_kv_cache: size = 100.00 MiB (1000000 cells,   2 layers,  8/1 seqs), K (q8_0): 50.00 MiB, V (q8_0): 50.00 MiB\n"
        ) + self.extra_log

    def follow_logs(self, name: str, path: Path) -> "FakeFollowing":
        path.parent.mkdir(parents=True, exist_ok=True)
        # Marked, so a test can tell the followed log from a dump that replaced it.
        path.write_text("(followed live)\n" + self.logs(name), encoding="utf-8")
        self.followed.append(name)
        return FakeFollowing()

    def exit_code(self, name: str) -> int | None:
        return None if name in self.running else 0

    def is_running(self, name: str) -> bool:
        return name in self.running

    def gguf_architectures(self, image: str) -> set[str] | None:
        """What the bench image's transformers reads. None (unknown) keeps the GGUF's own tokenizer."""
        return self.gguf_archs

    def list_sma(self) -> list[str]:
        return sorted(self.running | self.exited)


class FakeFollowing:
    def stop(self, timeout_s: float = 10.0) -> None:
        del timeout_s


def _backend(spec: StartSpec) -> str | None:
    argv = spec.serve_argv
    if "--moe-backend" not in argv:
        return None
    return argv[argv.index("--moe-backend") + 1]


class FakeEvaluator:
    """Writes the layout nemo-evaluator 0.3.0 writes, so SMA's own parsing runs in every pipeline test."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.configs: list[dict[str, Any]] = []
        self.client_timeouts: list[float | None] = []
        self.eval_factory_configs: list[dict[str, Any]] = []

    def run(
        self, argv: list[str], output_dir: Path, timeout_s: float, log_path: Path, client_timeout_s: float | None = None
    ) -> None:
        self.calls.append(list(argv))
        self.client_timeouts.append(client_timeout_s)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if "/app/llama-perplexity" in argv:
            self._perplexity(argv, log_path)
            return
        if argv[0] == "docker":
            self._eval_factory(argv)
            log_path.write_text("fake eval-factory\n", encoding="utf-8")
            return
        # nel runs in the suite folder: `nel eval run nel.yaml -o attempt-<n>`.
        config = yaml.safe_load((output_dir / argv[argv.index("run") + 1]).read_text(encoding="utf-8"))
        self.configs.append(config)
        bench = config["benchmarks"][0]
        out = output_dir / argv[argv.index("-o") + 1]
        bench_dir = out / bench["name"].replace("://", "___")
        bench_dir.mkdir(parents=True, exist_ok=True)
        count = bench["max_problems"] or 2
        # Real nel writes its absolute output path here, which can hold the hostname.
        (out / "nel_run.json").write_text(
            json.dumps({"output_dir": str(out.resolve()), "host": socket.gethostname()}), encoding="utf-8"
        )
        bundle = {
            "sdk_version": "0.3.0",
            "config": {"benchmark": bench["name"], "shuffle": {"seed": bench["shuffle_seed"], "ds_full_size": 541}},
            "benchmark": {
                "name": bench["name"],
                "samples": count,
                "repeats": 1,
                "scores": {
                    "pass@1": {"value": 0.5, "bootstrap_ci_lower": 0.25, "bootstrap_ci_upper": 0.75},
                    "failures": {"total_steps": count, "total_failures": 0},
                },
            },
        }
        (bench_dir / f"eval-20261027T045605Z-{bench_dir.name}.json").write_text(json.dumps(bundle), encoding="utf-8")
        rows = [
            {"problem_idx": index, "repeat": 0, "reward": float(index % 2 == 0), "trajectory": [{"source": "user"}]}
            for index in range(count)
        ]
        (bench_dir / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        log_path.write_text("fake nemo-evaluator\n", encoding="utf-8")

    @staticmethod
    def _perplexity(argv: list[str], log_path: Path) -> None:
        """llama-perplexity's real output (b11913), and the logits file a reference run writes."""
        data = Path(__file__).resolve().parent / "data" / "llamacpp"
        if "--kl-divergence" in argv:
            log_path.write_text((data / "perplexity-kld-b11913.log").read_text(encoding="utf-8"), encoding="utf-8")
            return
        log_path.write_text((data / "perplexity-base-b11913.log").read_text(encoding="utf-8"), encoding="utf-8")
        if "--kl-divergence-base" in argv:
            mounts = [argv[index + 1] for index, arg in enumerate(argv) if arg == "-v"]
            run_dir = Path(next(mount for mount in mounts if mount.endswith(":/sma-run:rw")).rsplit(":", 2)[0])
            (run_dir / "logprobs" / "kld-base.bin").write_bytes(b"logits")

    def _eval_factory(self, argv: list[str]) -> None:
        """The layout the eval-factory BFCL or LiveCodeBench container (26.03) writes into its /results mount."""
        mounts = [argv[index + 1] for index, arg in enumerate(argv) if arg == "-v"]
        results = Path(next(mount for mount in mounts if mount.endswith(":/results")).rsplit(":", 1)[0])
        config_mount = next(mount for mount in mounts if ":/config/" in mount)
        run_config = yaml.safe_load(Path(config_mount.split(":", 1)[0]).read_text(encoding="utf-8"))
        self.eval_factory_configs.append(run_config)
        if run_config["config"]["type"].startswith(("codegeneration", "livecodebench")):
            _livecodebench_output(results)
            return
        model = "org_model"
        (results / "result" / model).mkdir(parents=True, exist_ok=True)
        (results / "score" / model).mkdir(parents=True, exist_ok=True)
        categories = {"simple": (["simple_0", "simple_1", "simple_2"], ["simple_1"]), "multi_turn_base": (["base_0", "base_1"], [])}
        for category, (ids, failed) in categories.items():
            name = f"BFCL_v3_{category}"
            (results / "result" / model / f"{name}_result.json").write_text(
                "".join(json.dumps({"id": item, "result": []}) + "\n" for item in ids), encoding="utf-8"
            )
            summary = {"accuracy": 1 - len(failed) / len(ids), "correct_count": len(ids) - len(failed), "total_count": len(ids)}
            lines = [summary, *({"id": item, "valid": False} for item in failed)]
            (results / "score" / model / f"{name}_score.json").write_text(
                "".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8"
            )
        (results / "results.yml").write_text(
            yaml.safe_dump(
                {
                    "results": {
                        "groups": {
                            "bfcl": {"metrics": {"multi_turn_overall_acc": {"scores": {"multi_turn_overall_acc": {"value": 100.0}}}}}
                        }
                    }
                }
            ),
            encoding="utf-8",
        )


def _livecodebench_output(results: Path) -> None:
    """What livecodebench 26.03 writes for three problems with one answer each: two pass, one fails."""
    problems = [("abc390_a", "easy", True), ("abc390_f", "hard", False), ("3781", "medium", True)]
    rows = [
        {
            "question_id": qid, "difficulty": level, "platform": "atcoder", "contest_date": "2025-02-01T00:00:00",
            "output_list": [f"```python\nprint({qid!r})\n```"], "code_list": [f"print({qid!r})"],
            "graded_list": [ok], "pass@1": 1.0 if ok else 0.0,
        }
        for qid, level, ok in problems
    ]  # fmt: skip
    stem = results / "org_model" / "Scenario.codegeneration_1_0.0"
    stem.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{stem}_eval_all.json").write_text(json.dumps(rows), encoding="utf-8")
    Path(f"{stem}_eval.json").write_text(json.dumps([{"pass@1": 2 / 3, "pass@1_stderr": 0.27}]), encoding="utf-8")
    (results / "results.yml").write_text(
        yaml.safe_dump({"results": {"groups": {"codegeneration": {"metrics": {"pass@1": {"scores": {"pass@1": {"value": 2 / 3}}}}}}}}),
        encoding="utf-8",
    )


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
        prompts = 1
        if "--num-prompts" in argv:
            prompts = int(argv[argv.index("--num-prompts") + 1])
        target.write_text(
            json.dumps(
                {
                    "completed": prompts,
                    "errors": [],
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
