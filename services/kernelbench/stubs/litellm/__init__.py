"""Stand-in for LiteLLM (README.md). KernelBench's utils import it at load time for kernel
generation, which this service never does, so no model client is installed in an image that runs
model-written code."""


def completion(*args, **kwargs):
    raise RuntimeError("no model clients in the KernelBench evaluation image")
