# Benchmark services

SMA serves and measures one model on one Spark. Three benchmarks need another machine, and this folder holds what runs there. None is part of the `sma` package. Every `sma run` calls them, in its `kernelbench`, `swebench`, `browsecomp_plus`, `kernelbench_results`, and `browsecomp_plus_results` steps, with the machines named in the root `.env` (`.env.example` lists them). Each guide says how to deploy it and how to call it by hand.

| Service | Runs on | Why another machine | SMA reaches it | Guide |
| --- | --- | --- | --- | --- |
| KernelBench evaluation | A second NVIDIA GPU machine; the lab uses a DGX Spark | Compiling and timing kernels would disturb SMA's speed numbers, and model-written code should not run where models are served | HTTP, over SSH or a private network | [kernelbench/](kernelbench/README.md) |
| SWE-bench runner | An x86_64 Docker host | SWE-bench's task images are x86_64 | SSH, with a reverse forward to the model server | [swe-runner/](swe-runner/README.md) |
| BrowseComp-Plus search | Any x86_64 Linux machine; the lab uses the SWE-bench runner | The BM25 index and its search server would take memory and CPU from the Spark | SSH, with a forward from the Spark to the search server | [browsecomp-plus/](browsecomp-plus/README.md) |

The baselines measured on the lab's copies are in [`runs/gb10/baseline/kernelbench/`](../runs/gb10/baseline/kernelbench/README.md) and [`runs/cpu/baseline/sweverified/`](../runs/cpu/baseline/sweverified/README.md). A copy of your own gives other numbers: record its baseline before you compare anything.

`tests/test_services.py` checks the KernelBench server's request handling and results, and both services' pins, without a GPU or Docker. `tests/test_browsecomp.py` runs a whole BrowseComp-Plus run against a stand-in search server.

The judge that grades BrowseComp-Plus is not in this folder: it is the lab's Ornith 1.5 122B, described in [`judge-info.md`](../judge-info.md).
