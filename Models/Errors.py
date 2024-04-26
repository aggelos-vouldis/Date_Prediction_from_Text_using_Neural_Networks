class ImportNotFoundException(Exception):
    def __init__(self, message: str = "Import is not installed in your computer") -> None:
        super().__init__(message)


class DataWrongException(Exception):
    def __init__(self, message: str = "Data is not correct") -> None:
        super().__init__(message)
