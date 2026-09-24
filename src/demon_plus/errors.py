"""Expected, user-actionable pipeline failures with safe messages."""


class DataError(ValueError):
    """Input data violates the declared dataset contract."""
