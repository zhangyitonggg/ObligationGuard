class ObligationGuardError(Exception):
    """An input, configuration, or execution condition needs correction."""


class DataError(ObligationGuardError):
    pass


class ConfigurationError(ObligationGuardError):
    pass


class ModelOutputError(ObligationGuardError):
    pass
