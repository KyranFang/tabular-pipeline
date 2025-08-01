"""Tabular-LLM Pipeline (TLP) - Tabular data and large language model reasoning service

TLP is a Python library specifically designed for processing tabular data and reasoning through large language models.
It provides a complete data processing pipeline from file input to natural language query answering.

Main features:
- Multi-format tabular file input (CSV, Excel, Parquet, JSONL)
- Data normalization and preprocessing
- Natural language query processing based on large language models
- Flexible reasoning path selection
- Complete error handling and logging

Usage example:
    from tlp.pipeline import create_pipeline
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Process query
    result = pipeline.process_file("data.csv", "How many rows does this table have?")
    
    if result.success:
        print(result.answer)
    else:
        print(f"Processing failed: {result.error_message}")
    
    # Clean up resources
    pipeline.cleanup()
"""

__version__ = "0.1.0"
__author__ = "TLP Team"
__email__ = "tlp@example.com"
__description__ = "Tabular-LLM Pipeline - Tabular data and large language model reasoning service"

# Export main interfaces
from .pipeline import Pipeline, PipelineResult
from .exceptions import (
    TLPException,
    InputException,
    ProcessingException,
    ReasoningException,
    StorageException,
    ModelException,
    ValidationException,
    ConfigurationException
)

# Export utility functions
from .utils.logger import get_logger

__all__ = [
    # Core classes
    "Pipeline",
    "PipelineResult",
    
    # Exception classes
    "TLPException",
    "InputException",
    "ProcessingException",
    "ReasoningException",
    "StorageException",
    "ModelException",
    "ValidationException",
    "ConfigurationException",
    
    # Utility functions
    "get_logger",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]