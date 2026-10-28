"""Stand-in for the OpenAI client (README.md). KernelBench's utils import it at load time for
kernel generation, which this service never does; the real package isn't installed."""


class OpenAI:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("no model clients in the KernelBench evaluation image")
