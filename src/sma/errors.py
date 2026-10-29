"""User-facing errors. The CLI prints these and exits 2."""


class SmaError(Exception):
    """Expected failure with a message that is safe to show and to store."""


class QuantMismatch(SmaError):
    """The checkpoint's declared quantization does not match the flag."""


class CompareRefused(SmaError):
    """Two runs cannot be compared."""


class StepFailed(SmaError):
    def __init__(self, step: str, message: str) -> None:
        self.step = step
        super().__init__(f"{step}: {message}")


class RunIncomplete(SmaError):
    """The run went through every step, and some measurement or scoring steps failed. Resume retries them."""

    def __init__(self, run_id: str, steps: list[str]) -> None:
        self.run_id, self.steps = run_id, list(steps)
        super().__init__(
            f"Run {run_id} went through every step, but these failed: {', '.join(steps)}. Each error is in the "
            f"manifest. `sma resume {run_id}` retries them."
        )
