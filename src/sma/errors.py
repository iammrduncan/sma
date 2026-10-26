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
