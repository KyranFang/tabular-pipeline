import pandas as pd

from enum import Enum
from pydantic import BaseModel
from abc import ABC, abstractmethod

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from tlp.processing.base import ProcessingResult
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
    
class ReasoningResult(BaseModel):
    """Reasoning result"""
    answer: str
    reasoning_path: ReasoningPath
    prompts: List[Dict[str, str]] = []
    intermediate_results: List[Dict[str, Any]] = []
    success: bool = True
    error_message: Optional[str] = None   
    

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
    def reason(self, data: Any, query_info: ReasoningRequest) -> ReasoningResult:
        pass


class BaseExplainer(BaseReasoningOperator):
    """Base explainer class"""
    @abstractmethod
    def explain(self, reasoning_result: ReasoningResult) -> str:
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.model_name,
            'type': self.__class__.__name__,
            'loaded': self.is_loaded(),
            'config': self.config
        }