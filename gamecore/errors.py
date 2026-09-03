"""GameCore exception types."""


class GameCoreError(Exception):
    """Base error for deterministic benchmark execution."""


class ContextValidationError(GameCoreError):
    """Raised when a GameContext does not satisfy its contract."""


class ActionValidationError(GameCoreError):
    """Raised when an action or its preconditions are invalid."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
