import json
import pandas as pd

from typing import List, Dict, Optional, Any, Union

from tlp.reasoning.models.local_model import LocalModel
from tlp.exceptions import ReasoningException, ModelException
from tlp.reasoning.base import BaseReasoner, BaseQueryProcessor, QueryType, ReasoningPath, ReasoningResult, ReasoningRequest
from tlp.utils.utils import dataframe_to_string
from config.settings import settings
from config.model_config import get_model_config

class SimpleReasoner(BaseReasoner):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        self.model_name = config.get("model_name", "qwen2.5-7b")
        self.data_path = config.get("data_path", "")
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
            
    def _load_data(self):
        if not self.data_path:
            logger.error("Data path is not specified")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            record = json.loads(line)
            data_json_str = record.get('data', '')
            if not data_json_str:
                raise ReasoningException("No data field found in jsonl record")
            data_records = json.loads(data_json_str)
            
            return df
    
    def validate_input(self, data: pd.DataFrame, query: str) -> bool:
        """Validate input data and query"""
        if data is None or data.empty:
            return False
        if not query or not query.strip():
            return False
        return True
    
    def _decide_reasoning_plan(self, reasoning_request: ReasoningRequest, data: Union[pd.DataFrame, str]) -> ReasoningRequest:
        # we do not conduct special practice to the request here since it is a MVP version
        reasoning_request.reasoning_path = [ReasoningPath.DIRECT_REASONING]
        assert len(reasoning_request.sub_queries) == len(reasoning_request.reasoning_path)
        return reasoning_request
    
    def _execute_reasoning_plan(self, reasoning_request: ReasoningRequest, data: Union[pd.DataFrame, str]) -> ReasoningResult:
        answers = ''
        intermediate_answers = []
        prompts = self._generate_prompt(reasoning_request, data)
        for i in range(len(prompts)):
            intermediate_answers.append(self.model.generate(prompts[i]))
        
        answers = self._aggregate_answers(intermediate_answers)
        return ReasoningResult(
            answer=answers,
            reasoning_path=reasoning_request.reasoning_path[0],
            intermediate_results=[{"answer": ans} for ans in intermediate_answers]
        )
    
    def _generate_prompt(self, reasoning_request: ReasoningRequest, data: Union[pd.DataFrame, str]) -> List[str]:
        assert len(reasoning_request.reasoning_path) == 1
        reasoning_path = reasoning_request.reasoning_path[0]
        assert reasoning_path == ReasoningPath.DIRECT_REASONING
        
        if isinstance(data, pd.DataFrame):
            data = dataframe_to_string(data)
        else:
            data = data
        
        prompt = f"""You are a professional data analysis assistant. Please answer the user's question based on the following table data.
                    
                    Table data:
                    {data}

                    User question: {reasoning_request.query}

                    Please carefully analyze the table data and provide accurate, detailed answers. If calculations are needed, please explain the calculation process. If the data is insufficient to answer the question, please clearly state so.

                    Answer:"""
        
        return [prompt]
    
    def _aggregate_answers(self, intermediate_answers: List[str]) -> str:
        return intermediate_answers[0] # only in this toy case since we assert there is only sub query thus one inter_answer
    
    def reason(self, data: Union[pd.DataFrame, str, None] = None, query: str = "") -> ReasoningResult:
        # If no data is provided, load from data_path
        if data is None:
            data = self._load_data()
            
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