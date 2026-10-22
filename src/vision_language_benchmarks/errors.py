class BenchmarkError(Exception):
    """Base exception for benchmark failures."""


class DependencyUnavailableError(BenchmarkError):
    """A required optional model dependency is not installed."""


class ModelAccessBlockedError(BenchmarkError):
    """Model inference is blocked by an unresolved access or license condition."""


class DatasetUnavailableError(BenchmarkError):
    """A dataset is not available for legal or technical reasons."""


class DataIntegrityError(BenchmarkError):
    """A download or prepared dataset failed integrity validation."""


class ResumeMismatchError(BenchmarkError):
    """A previous run does not match the requested configuration."""


class BatchSizeExhaustedError(BenchmarkError):
    def __init__(self, message: str, oom_history: list[dict[str, object]]) -> None:
        super().__init__(message)
        self.oom_history = oom_history
