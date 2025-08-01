import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from tlp.input.file_uploader import FileUploader
from tlp.input.base import InputResult

from tlp.processing.basic_normalizer import DataNormalizer
from tlp.processing.base import ProcessingResult

from tlp.reasoning.basic_reasoner import SimpleReasoner
from tlp.reasoning.base import ReasoningResult

from tlp.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

class PipelineResult:
    def __init__(self):
        self.trace_id: Optional[str] = None
        self.success: bool = False
        self.error_message: Optional[str] = None
        self.input_result: Optional[InputResult] = None
        self.processing_results: List[ProcessingResult] = []
        self.reasoning_result: Optional[ReasoningResult] = None
        self.answer: Optional[str] = None
        self.explanation: Optional[str] = None
        self.total_execution_time: float = 0.0
        self.pipeline_config: Dict[str, Any] = {}
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'success': self.success,
            'error_message': self.error_message,
            'answer': self.answer,
            'explanation': self.explanation,
            'total_execution_time': self.total_execution_time,
            'input_metadata': self.input_result.metadata.dict() if self.input_result else None,
            'processing_steps': len(self.processing_results),
            'reasoning_path': self.reasoning_result.reasoning_path if self.reasoning_result else None
        }



class Pipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._init_components()
        
    def _init_components(self):
        input_config = self.config.get('input', {})
        self.file_uploader = FileUploader(input_config.get('file_uploader', {}))
        
        processing_config = self.config.get('processing', {})
        self.normalizer = DataNormalizer(processing_config.get('normalizer', {}))
        
        reasoning_config = self.config.get('reasoning', {})
        self.reasoner = SimpleReasoner(reasoning_config.get('reasoner', {}))

    def _process_input(self, file_path: Path):
        return self.file_uploader.process(file_path)
    
    def _process_data(self, intermediate_result: Any, output_file_path: Union[str, Path]):
        process_operators = [("normalizer", self.normalizer),
                             ("cleaner", None)]
        process_results = []
        final_process_operator = None
        
        for op_name, process_operator in process_operators:
            if process_operator is None:
                logger.warning(f"{op_name} is None, skip this step.")
                continue
            
            final_process_operator = process_operator
            if hasattr(process_operator, 'should_skip') and process_operator.should_skip():
                logger.info(f"{op_name} is skipped as required.")
                continue
                
            process_result = process_operator.process(intermediate_result)
            
            if process_result.success:
                process_results.append(process_result)
                intermediate_result = process_result
            else:
                logger.error(f"{op_name} failed with error: {process_result.error_message}")
                break
        
        if final_process_operator is None:
            logger.warning("No valid process operator found.")
        final_process_operator.save_result(intermediate_result, output_file_path)
        return process_results

    def _process_reasoning(self, data: pd.DataFrame, query: str):
        reasoning_result = self.reasoner.reason(data, query)
        if reasoning_result.answer == "":
            logger.warning("Reasoning result is empty.")
        return reasoning_result

    def process(self, input_file_path: Union[str, Path], output_file_path: Union[str, Path], query: str, **kwargs):
        input_file_path = Path(input_file_path)
        output_file_path = Path(output_file_path)
        result = PipelineResult() 
        
        result.input_result = self._process_input(input_file_path)
        
        # Check if input processing was successful
        if not result.input_result.success or result.input_result.data is None:
            logger.error(f"Input processing failed: {result.input_result.error_message}")
            raise Exception(f"Input processing failed: {result.input_result.error_message}")
        
        result.processing_results = self._process_data(result.input_result.data, output_file_path)
        
        # Check if processing was successful
        if not result.processing_results or not result.processing_results[-1].success:
            error_msg = result.processing_results[-1].error_message if result.processing_results else "No processing results"
            logger.error(f"Data processing failed: {error_msg}")
            raise Exception(f"Data processing failed: {error_msg}")
            
        processed_data = result.processing_results[-1].data
        result.reasoning_result = self._process_reasoning(processed_data, query)
        result.answer = result.reasoning_result.answer
        return result