class PanicError(Exception):
    """If this error raised, render process should stop because some non-random error has occurred"""

    pass


class RetryError(Exception):
    pass
