# Benchmark services

SMA serves and measures one model on one Spark. Three benchmarks need another machine, and this folder holds what runs there. None is part of the `sma` package. Every `sma run` calls them, in its `kernelbench`, `swebench`, `browsecomp_plus`, `kernelbench_results`, and `browsecomp_plus_results` steps, with the machines named in the root `.env` (`.env.example` lists them). Each guide says how to deploy it and how to call it by hand.

| Service | Runs on | Why another machine | SMA reaches it | Guide |
| --- | --- | --- | --- | --- |
| KernelBench evaluation | An NVIDIA GPU machine that serves no model; since 2026-11-01 the lab runs it on the harness GB10, isolated from the harness by `host-setup.sh` | Compiling and timing kernels would disturb a model server's speed numbers, and model-written code should not run where models are served. The models are on the V100 pod now | HTTP: on the GB10's internal Docker network, or over SSH from another machine | [kernelbench/](kernelbench/README.md) |
| SWE-bench runner | An x86_64 Docker host | SWE-bench's task images are x86_64 | SSH, with a reverse forward to the model server | [swe-runner/](swe-runner/README.md) |
| BrowseComp-Plus search | Any x86_64 Linux machine; the lab uses the SWE-bench runner | The BM25 index and its search server would take memory and CPU from the Spark | SSH, with a forward from the Spark to the search server | [browsecomp-plus/](browsecomp-plus/README.md) |
| V100 serving | The lab's V100 pod | The V100s serve GGUF models on llama.cpp, one run per GPU (`sma worker --slots`); the harness stays on the Spark | SSH, with local forwards to the model ports | [v100-serving/](v100-serving/README.md) |

The baselines measured on the lab's copies are in [`runs/gb10/baseline/kernelbench/`](../runs/gb10/baseline/kernelbench/README.md) and [`runs/cpu/baseline/sweverified/`](../runs/cpu/baseline/sweverified/README.md). A copy of your own gives other numbers: record its baseline before you compare anything.

`tests/test_services.py` checks the KernelBench server's request handling and results, and both services' pins, without a GPU or Docker. `tests/test_browsecomp.py` runs a whole BrowseComp-Plus run against a stand-in search server.

The judge that grades BrowseComp-Plus is not in this folder: since 2026-11-01 it is Qwen 3.8 Flash (FP4) behind an OpenAI-compatible endpoint (`browsecomp.judge_model`); before, Ornith 1.5 122B, described in [`judge-info.md`](../judge-info.md).
