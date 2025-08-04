from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from enum import Enum
from pydantic import Field

from tlp.data_structure.base import BaseModuleOutput, BaseMetadata


class QueryType(Enum):
    """Query type enumeration"""
    SIMPLE = "simple"
    BENCHMARK = "benchmark"
    COMPLEX = "complex"


class QueryMetadata(BaseMetadata):
    """Query metadata"""
    query_type: QueryType
    original_format: str
    processed_format: str
    validation_passed: bool
    enhancement_applied: bool = False
    

class QueryOutput(BaseModuleOutput):
    """Query processing output"""
    metadata: QueryMetadata = Field(alias='_metadata')
    
    class Config:
        populate_by_name = True
        validate_by_name = True
    
    def __init__(self, data: Any, metadata: QueryMetadata, success: bool = True, error_message: Optional[str] = None):
        super().__init__(
            _data=data,
            _metadata=metadata,
            _success=success,
            _error_message=error_message
        )


class BaseQueryProcessor(ABC):
    """Base query processor"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def extract_query(self, data: Any) -> Union[str, List[str]]:
        """Extract query from input data"""
        pass
    
    @abstractmethod
    def normalize_query(self, query: Union[str, List[str]]) -> Union[str, List[str]]:
        """Normalize query format"""
        pass
    
    @abstractmethod
    def classify_query(self, query: Union[str, List[str]]) -> QueryType:
        """Classify query type"""
        pass
    
    def validate_query(self, query: Union[str, List[str]]) -> bool:
        """Validate query"""
        if isinstance(query, str):
            return len(query.strip()) > 0
        elif isinstance(query, list):
            return len(query) > 0 and all(len(q.strip()) > 0 for q in query)
        return False
    
    def enhance_query(self, query: Union[str, List[str]], context: Optional[Dict[str, Any]] = None) -> Union[str, List[str]]:
        """Enhance query with context"""
        return query
    
    def process(self, data: Any, context: Optional[Dict[str, Any]] = None) -> QueryOutput:
        """Process query"""
        # Extract query
        query = self.extract_query(data)
        
        # Normalize query
        normalized_query = self.normalize_query(query)
        
        # Classify query
        query_type = self.classify_query(normalized_query)
        
        # Validate query
        validation_passed = self.validate_query(normalized_query)
        
        # Enhance query
        enhanced_query = self.enhance_query(normalized_query, context)
        
        # Create metadata
        metadata = QueryMetadata(
            query_type=query_type,
            original_format=str(type(query).__name__),
            processed_format=str(type(enhanced_query).__name__),
            validation_passed=validation_passed,
            enhancement_applied=enhanced_query != normalized_query
        )
        
        return QueryOutput(enhanced_query, metadata)