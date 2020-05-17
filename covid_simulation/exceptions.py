# define Python user-defined exceptions
class Error(Exception):
    """Base class for other exceptions"""
    pass


class DataFormatChangeError(Error):
    """Downloaded data has change its format.

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class FileDownloadError(Error):
    """
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class DistributionError(Error):
    """
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class FailedOptimizationError(Error):
    """
    Failed to converge in optimization algorithm
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class ModelNotDefinedError(Error):
    """
    Model has not been defined.
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class ParameterNumberError(Error):
    """
    The number of parameters is not correct
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class DicModelError(Error):
    """
    model dictionary not well defined
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message
