"""Custom exception module"""


class TLPException(Exception):
    """TLP base exception class"""
    pass


class InputException(TLPException):
    """Input layer exception"""
    pass


class FileFormatException(InputException):
    """File format exception"""
    pass


class FileSizeException(InputException):
    """File size exception"""
    pass


class ProcessingException(TLPException):
    """Processing layer exception"""
    pass


class DetectionException(ProcessingException):
    """Table detection exception"""
    pass


class NormalizationException(ProcessingException):
    """Normalization exception"""
    pass


class CleaningException(ProcessingException):
    """Cleaning exception"""
    pass


class FeatureException(ProcessingException):
    """Feature engineering exception"""
    pass


class ReasoningException(TLPException):
    """Reasoning layer exception"""
    pass


class QueryException(ReasoningException):
    """Query processing exception"""
    pass


class ModelException(ReasoningException):
    """Model exception"""
    pass


class ModelLoadException(ModelException):
    """Model loading exception"""
    pass


class InferenceException(ModelException):
    """Inference exception"""
    pass


class StorageException(TLPException):
    """Storage layer exception"""
    pass


class MetadataException(StorageException):
    """Metadata exception"""
    pass


class ValidationException(TLPException):
    """Data validation exception"""
    pass


class ConfigurationException(TLPException):
    """Configuration exception"""
    pass