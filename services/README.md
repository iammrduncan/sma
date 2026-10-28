# Benchmark services

SMA serves and measures one model on one Spark. Two benchmarks need another machine, and this folder holds what runs there. Neither is part of the `sma` package, and SMA does not call either one yet. Each guide says how to deploy it and how to call it during an eval.

| Service | Runs on | Why another machine | SMA reaches it | Guide |
| --- | --- | --- | --- | --- |
| KernelBench evaluation | A second NVIDIA GPU machine; the lab uses a DGX Spark | Compiling and timing kernels would disturb SMA's speed numbers, and model-written code should not run where models are served | HTTP, over SSH or a private network | [kernelbench/](kernelbench/README.md) |
| SWE-bench runner | An x86_64 Docker host | SWE-bench's task images are x86_64 | SSH, with a reverse forward to the model server | [swe-runner/](swe-runner/README.md) |

The baselines measured on the lab's copies are in [`runs/gb10/baseline/kernelbench/`](../runs/gb10/baseline/kernelbench/README.md) and [`runs/cpu/baseline/sweverified/`](../runs/cpu/baseline/sweverified/README.md). A copy of your own gives other numbers: record its baseline before you compare anything.

`tests/test_services.py` checks the KernelBench server's request handling and results, and both services' pins, without a GPU or Docker.
