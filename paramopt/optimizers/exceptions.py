class ParameterError(Exception):
    """プロセスパラメータに関する例外クラス."""
    def __init__(self, message):
        super().__init__(message)


class FittingError(Exception):
    """学習に関する例外クラス."""
    def __init__(self, message):
        super().__init__(message)
