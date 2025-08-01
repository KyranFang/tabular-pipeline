import pandas as pd

from enum import Enum
from pydantic import BaseModel, Field
import uuid
from abc import ABC, abstractmethod

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from tlp.data_structure import BaseMetadata, BaseModuleOutput
from tlp.exceptions import ReasoningException
from tlp.utils.logger import get_logger

logger = get_logger(__name__)

class QueryType(str, Enum):
    SIMPLE = "simple"
    STATISTICAL = "statistical"
    FILTER = "filter"
    PREDICTION = "prediction"
    GENERATION = "generation"
    COMPARISON = "comparison"
    ANOMALY = "anomaly"
    COMPLEX = "complex"

class ReasoningPath(str, Enum):
    """Reasoning path enumeration"""
    DIRECT_REASONING = "direct_reasoning"
    SQL_TOOL = "sql_tool"
    PYTHON_TOOL = "python_tool"
    HYBRID = "hybrid"

class ReasoningRequest(BaseModel):
    query: str
    query_type: Optional[QueryType] = None
    sub_queries: List[str] = []
    sub_query_types: List[QueryType] = []
    aggregation_needed: bool = False
    table_info: Dict[str, Any] = {}
    expected_columns: List[str] = []
    tool_code: Optional[str] = None
    reasoning_path: List[ReasoningPath] = []
    
class ReasoningMetadata(BaseMetadata):
    reasoning_path: ReasoningPath = ReasoningPath.DIRECT_REASONING
    prompts: Union[str, List[Dict[str, str]]] = []

class ReasoningOutput(BaseModuleOutput):
    """Reasoning result"""
    # Inherits all fields from BaseModuleOutput
    # Override metadata type to be more specific
    metadata: ReasoningMetadata = Field(alias='_metadata')
    
    class Config:
        populate_by_name = True
        validate_by_name = True
    
    @property
    def _answer(self) -> str:
        """Backward compatibility property for answer access"""
        return self.data if isinstance(self.data, str) else str(self.data)   
    
class BaseReasoningOperator(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        self.operator_name = self.__class__.__name__
    
    @abstractmethod
    def validate_input(self, data: Any, query: str) -> bool:
        pass
    
    def is_enabled(self) -> bool:
        """Check if operator is enabled"""
        return self.config.get('enabled', True)


class BaseQueryProcessor(BaseReasoningOperator):
    def __init__(self, config):
        super().__init__(config)
    
    @abstractmethod
    def process(self, query: str) -> ReasoningRequest:
        pass
    
    def _query_classification(self, query: str) -> QueryType:
        pass
    
    def _query_decomposition(self, query: str) -> List[str]:
        pass


class BaseReasoner(BaseReasoningOperator):
    @abstractmethod
    def reason(self, data: Any, query_info: ReasoningRequest) -> ReasoningOutput:
        pass


class BaseExplainer(BaseReasoningOperator):
    """Base explainer class"""
    @abstractmethod
    def explain(self, reasoning_result: ReasoningOutput) -> str:
        pass


class BaseModel(ABC):
    """Base model class"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        self.model_name = config.get('name', 'unknown')
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load model"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass