import json
import pandas as pd

from pathlib import Path
from typing import List, Dict, Optional, Any, Union


from tlp.reasoning.models.local_model import LocalModel
from tlp.exceptions import ReasoningException, ModelException
from tlp.reasoning.base import BaseReasoner, BaseQueryProcessor, QueryType, ReasoningPath, ReasoningOutput, ReasoningRequest
from tlp.utils.utils import dataframe_to_string
from config.settings import settings
from config.model_config import get_model_config

class SimpleReasoner(BaseReasoner):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = config.get("model_name", "qwen2.5-7b")
        self.model = None
        self.query_processor = None
        
        self._load_model()
        self._load_query_processor()
        
    def _load_model(self):
        if self.model is None:
            model_config = get_model_config(self.model_name)
            self.model = LocalModel(model_config)
    
    def _load_query_processor(self):
        if self.query_processor is None:
            self.query_processor = SimpleQueryProcessor()
    
    def validate_input(self, data: pd.DataFrame, query: str) -> bool:
        """Validate input data and query"""
        if data is None or data.empty:
            return False
        if not query or not query.strip():
            return False
        return True
    
    def _decide_reasoning_plan(self, reasoning_request: ReasoningRequest, data: str) -> ReasoningRequest:
        # we do not conduct special practice to the request here since it is a MVP version
        reasoning_request.reasoning_path = [ReasoningPath.DIRECT_REASONING]
        assert len(reasoning_request.sub_queries) == len(reasoning_request.reasoning_path)
        return reasoning_request
    
    def _execute_reasoning_plan(self, reasoning_request: ReasoningRequest, data: str) -> ReasoningOutput:
        answers = ''
        intermediate_answers = []
        prompts = self._generate_prompt(reasoning_request, data)
        for i in range(len(prompts)):
            intermediate_answers.append(self.model.generate(prompts[i]))
        
        answers = self._aggregate_answers(intermediate_answers)
        
        # Create metadata
        from tlp.reasoning.base import ReasoningMetadata
        metadata = ReasoningMetadata(
            reasoning_path=reasoning_request.reasoning_path[0],
            prompts=prompts[0] if prompts else ""  # Use first prompt as string
        )
        
        return ReasoningOutput(
            data=answers,
            metadata=metadata,
            success=True
        )
    
    def _generate_prompt(self, reasoning_request: ReasoningRequest, data: str) -> List[str]:
        assert len(reasoning_request.reasoning_path) == 1 # since this is a simple reasoner
        reasoning_path = reasoning_request.reasoning_path[0]
        assert reasoning_path == ReasoningPath.DIRECT_REASONING
        
        prompt = f"""You are a professional data analysis assistant. Please answer the user's question based on the following table data.
                    
                    Table data:
                    {data}

                    User question: {reasoning_request.query}

                    Please carefully analyze the table data and only give me the answer without any explanation. If the data is insufficient to answer the question, please clearly state so.

                    Answer:"""
        
        return [prompt]
    
    def _aggregate_answers(self, intermediate_answers: List[str]) -> str:
        return intermediate_answers[0] # only in this toy case since we assert there is only sub query thus one inter_answer
    
    def _load_data(self, data_path: Union[str, Path]) -> str:
        if not data_path:
            logger.error("Data path is not specified")
        with open(data_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            record = json.loads(line)
            data_json_str = record.get('data', '')
            if not data_json_str:
                raise ReasoningException("No data field found in jsonl record")
            
            return data_json_str
    
    def _process_input(self, input_data) -> str:
        """should return a string of table"""
        if isinstance(input_data, pd.DataFrame):
            return dataframe_to_string(input_data)
        elif isinstance(input_data, str) or isinstance(input_data, Path):
            return self._load_data(input_data)
        elif isinstance(input_data, ProcessingResult):
            return input_data.data
        else:
            raise ReasoningException("Invalid input data type")
    
    def reason(self, input_data: Any = None, query: str = "") -> ReasoningOutput:
        """input_data could be any of {pd.DataFrame, str, Path, ProcessingResult}
           if isinstance(input_data, ProcessingResult):
                this case is an pure online case, user input a table and a query, do not require re-use the table (or maybe the first-time inference)
           elif isinstance(input_data, str) or isinstance(input_data, Path):
                this case is reusing a pre-uploaded table
           elif isinstance(input_data, pd.DataFrame):
                just in case, it should not be the standard use
        """
        
        data = self._process_input(input_data)
            
        reasoning_request = self.query_processor.process(query)
        reasoning_request = self._decide_reasoning_plan(reasoning_request, data)
        
        reasoning_result = self._execute_reasoning_plan(reasoning_request, data)
        
        return reasoning_result
    
class SimpleQueryProcessor(BaseQueryProcessor):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        config = config or {}
        self.model_name = config.get("model_name", "qwen2.5-7b")
        
    def validate_input(self, data: pd.DataFrame, query: str) -> bool:
        """Validate input data and query"""
        if not query or not query.strip():
            return False
        return True
        
    def _query_classification(self, query: str) -> QueryType:
        return QueryType.SIMPLE
    
    def _query_decomposition(self, query: str) -> List[str]:
        return [query]
    
    def process(self, query: str) -> ReasoningRequest:
        query_type = self._query_classification(query)
        if query_type == QueryType.COMPLEX:
            sub_queries = self._query_decomposition(query)
            sub_query_types = [self._query_classification(sub_query) for sub_query in sub_queries]
        else:
            sub_queries = [query]
            sub_query_types = [query_type]
        return ReasoningRequest(
            query=query,
            query_type=query_type,
            sub_queries=sub_queries,
            sub_query_types=sub_query_types
        )