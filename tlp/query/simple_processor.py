from typing import Dict, List, Any, Optional, Union
from .base import BaseQueryProcessor, QueryType


class SimpleQueryProcessor(BaseQueryProcessor):
    """Simple query processor for basic string queries"""
    
    def extract_query(self, data: Any) -> str:
        """Extract query from input data"""
        if isinstance(data, str):
            return data
        elif isinstance(data, dict) and 'query' in data:
            return data['query']
        elif hasattr(data, 'query'):
            return data.query
        else:
            raise ValueError(f"Cannot extract query from data type: {type(data)}")
    
    def normalize_query(self, query: str) -> str:
        """Normalize query format"""
        if not isinstance(query, str):
            query = str(query)
        
        # Basic normalization
        query = query.strip()
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        return query
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query type"""
        # Simple heuristics for classification
        if len(query.split()) <= 5:
            return QueryType.SIMPLE
        else:
            return QueryType.COMPLEX
    
    def enhance_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Enhance query with context"""
        if context is None:
            return query
        
        # Add table context if available
        if 'table_info' in context:
            table_info = context['table_info']
            if isinstance(table_info, dict) and 'columns' in table_info:
                columns = table_info['columns']
                enhanced_query = f"Given table with columns {columns}, {query}"
                return enhanced_query
        
        return query