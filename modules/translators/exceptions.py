class BaseError(Exception):
    """
    Base error structure class for translators
    """

    def __init__(self, val, message):
        self.val = val
        self.message = message
        super().__init__()

    def __str__(self):
        return f"{self.val} --> {self.message}"


class InvalidSourceOrTargetLanguage(BaseError):
    def __init__(self, val, message="source and target language can't be the same"):
        super().__init__(val, message)


class TranslatorSetupFailure(Exception):
    pass


class MissingTranslatorParams(Exception):
    pass


class TranslatorNotValid(Exception):
    pass