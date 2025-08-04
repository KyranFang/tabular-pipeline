from typing import Dict, List, Any, Optional, Union
from .base import BaseQueryProcessor, QueryType


class BenchmarkQueryProcessor(BaseQueryProcessor):
    """Benchmark query processor for FinQA, TART, TableBench datasets"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.dataset_type = self.config.get('dataset_type', 'finqa')
        self.query_field_mapping = self.config.get('query_field_mapping', {
            'finqa': 'qa.question',
            'tart': 'question', 
            'tablebench': 'instruction'
        })
    
    def extract_query(self, data: Any) -> Union[str, List[str]]:
        """Extract query from benchmark data"""
        if isinstance(data, dict):
            return self._extract_from_dict(data)
        elif isinstance(data, list):
            # Handle batch of samples
            return [self._extract_from_dict(item) for item in data]
        else:
            raise ValueError(f"Cannot extract query from data type: {type(data)}")
    
    def _extract_from_dict(self, data: Dict[str, Any]) -> str:
        """Extract query from dictionary based on dataset type"""
        field_path = self.query_field_mapping.get(self.dataset_type, 'question')
        
        # Handle nested field paths like 'qa.question'
        if '.' in field_path:
            keys = field_path.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    raise KeyError(f"Field path '{field_path}' not found in data")
            return value
        else:
            if field_path in data:
                return data[field_path]
            else:
                raise KeyError(f"Field '{field_path}' not found in data")
    
    def normalize_query(self, query: Union[str, List[str]]) -> Union[str, List[str]]:
        """Normalize query format"""
        if isinstance(query, str):
            return self._normalize_single_query(query)
        elif isinstance(query, list):
            return [self._normalize_single_query(q) for q in query]
        else:
            return str(query)
    
    def _normalize_single_query(self, query: str) -> str:
        """Normalize single query"""
        if not isinstance(query, str):
            query = str(query)
        
        # Basic normalization
        query = query.strip()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Dataset-specific normalization
        if self.dataset_type == 'finqa':
            # FinQA specific normalization
            query = query.replace('$', 'dollar ')
            query = query.replace('%', ' percent')
        elif self.dataset_type == 'tablebench':
            # TableBench specific normalization
            if query.startswith('Instruction: '):
                query = query[13:]  # Remove 'Instruction: ' prefix
        
        return query
    
    def classify_query(self, query: Union[str, List[str]]) -> QueryType:
        """Classify query type"""
        if isinstance(query, list):
            return QueryType.BENCHMARK
        
        # Benchmark queries are typically complex
        if self.dataset_type in ['finqa', 'tart', 'tablebench']:
            return QueryType.BENCHMARK
        
        # Fallback classification
        if len(query.split()) > 10:
            return QueryType.COMPLEX
        else:
            return QueryType.SIMPLE
    
    def enhance_query(self, query: Union[str, List[str]], context: Optional[Dict[str, Any]] = None) -> Union[str, List[str]]:
        """Enhance query with context"""
        if context is None:
            return query
        
        if isinstance(query, str):
            return self._enhance_single_query(query, context)
        elif isinstance(query, list):
            return [self._enhance_single_query(q, context) for q in query]
        else:
            return query
    
    def _enhance_single_query(self, query: str, context: Dict[str, Any]) -> str:
        """Enhance single query with context"""
        enhanced_query = query
        
        # Add table context
        if 'table_info' in context:
            table_info = context['table_info']
            if isinstance(table_info, dict):
                if 'columns' in table_info:
                    columns = table_info['columns']
                    enhanced_query = f"Given table with columns {columns}, {enhanced_query}"
                
                if 'description' in table_info:
                    description = table_info['description']
                    enhanced_query = f"{description}\n{enhanced_query}"
        
        # Add pre/post text context for FinQA
        if self.dataset_type == 'finqa' and 'pre_text' in context:
            pre_text = context['pre_text']
            enhanced_query = f"Context: {pre_text}\nQuestion: {enhanced_query}"
        
        return enhanced_query