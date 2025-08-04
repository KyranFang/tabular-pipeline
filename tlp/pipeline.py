import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from tlp.input.file_uploader import FileUploader
from tlp.input.dataset_uploader import DatasetUploader
from tlp.input.base import FileUploadeOutput

from tlp.processing.basic_normalizer import DataNormalizer
from tlp.processing.base import ProcessingOutput

from tlp.query import QueryOutput
from tlp.reasoning.basic_reasoner import SimpleReasoner, SimpleQueryProcessor
from tlp.reasoning.base import ReasoningOutput

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
        self.reasoning_result: Optional[ReasoningOutput] = None
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
            'reasoning_path': self.reasoning_result.metadata.reasoning_path if self.reasoning_result else None
        }



class Pipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._init_components()
        
    def _init_components(self):
        input_config = self.config.get('input', {})
        self.file_uploader = FileUploader(input_config.get('file_uploader', {}))
        self.dataset_uploader = DatasetUploader(input_config.get('dataset_uploader', {}))
        
        processing_config = self.config.get('processing', {})
        self.normalizer = DataNormalizer(processing_config.get('normalizer', {}))
        
        query_config = self.config.get('query', {})
        self.simple_query_processor = SimpleQueryProcessor(query_config.get('simple', {}))
        # For now, use SimpleQueryProcessor for both simple and benchmark modes
        # since they both handle string queries in the reasoning context
        self.benchmark_query_processor = SimpleQueryProcessor(query_config.get('benchmark', {}))
        
        reasoning_config = self.config.get('reasoning', {})
        query_processor_type = reasoning_config.get('query_processor_type', 'simple')
        if query_processor_type == 'benchmark':
            query_processor = self.benchmark_query_processor
        else:
            query_processor = self.simple_query_processor
        self.reasoner = SimpleReasoner(reasoning_config.get('reasoner', {}), query_processor=query_processor)

    def _process_input(self, file_path: Path):
        return self.file_uploader.process(file_path)
    
    def _process_dataset_input(self, file_path: Path):
        """Process benchmark dataset input"""
        return self.dataset_uploader.process(file_path)
    
    def _process_query(self, query_data: Any, is_benchmark: bool = False, context: Optional[Dict[str, Any]] = None) -> QueryOutput:
        """Process query using appropriate processor"""
        # Note: reasoning module query processors only accept query string
        # For now, we assume query_data is a string for both simple and benchmark modes
        if isinstance(query_data, str):
            query_str = query_data
        else:
            # If query_data is not a string, we need to extract the query
            # This is a temporary solution - ideally we should have proper query extraction
            query_str = str(query_data)
        
        if is_benchmark:
            reasoning_request = self.benchmark_query_processor.process(query_str)
        else:
            reasoning_request = self.simple_query_processor.process(query_str)
        
        # Convert ReasoningRequest to QueryOutput for compatibility
        from tlp.query.base import QueryMetadata, QueryType as QueryTypeEnum
        
        # Map reasoning QueryType to query QueryType
        query_type_mapping = {
            'simple': QueryTypeEnum.SIMPLE,
            'benchmark': QueryTypeEnum.BENCHMARK,
            'complex': QueryTypeEnum.COMPLEX
        }
        
        query_type = query_type_mapping.get(reasoning_request.query_type.value, QueryTypeEnum.SIMPLE)
        
        metadata = QueryMetadata(
            query_type=query_type,
            original_format='string',
            processed_format='string',
            validation_passed=True,
            enhancement_applied=False,
            processing_time=0.0
        )
        
        return QueryOutput(
            data=reasoning_request.query,
            metadata=metadata
        )
    
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

    def _process_reasoning_from_data(self, data: pd.DataFrame, query: str):
        """Process reasoning directly from DataFrame"""
        reasoning_result = self.reasoner.reason(data, query)
        if reasoning_result.answer == "":
            logger.warning("Reasoning result is empty.")
        return reasoning_result
    
    def _process_reasoning_from_file(self, file_path: Union[str, Path], query: str):
        """Process reasoning from saved jsonl file"""
        reasoning_result = self.reasoner.reason(file_path, query)
        if reasoning_result.answer == "":
            logger.warning("Reasoning result is empty.")
        return reasoning_result

    def process(self, input_file_path: Union[str, Path], output_file_path: Union[str, Path], query: str, use_saved_data: bool = False, **kwargs):
        """Process pipeline with option to use saved data for reasoning
        
        Args:
            input_file_path: Path to input file
            output_file_path: Path to save processed data
            query: Query for reasoning
            use_saved_data: If True, reasoning will read from saved jsonl file; if False, use processed data directly
        """
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
            
        # Choose reasoning mode based on use_saved_data flag
        if use_saved_data:
            # Read from saved jsonl file
            result.reasoning_result = self._process_reasoning_from_file(output_file_path, query)
        else:
            # Use processed data directly
            processed_data = result.processing_results[-1].data
            result.reasoning_result = self._process_reasoning_from_data(processed_data, query)
            
        result.answer = result.reasoning_result.answer
        
        # Set overall success based on all components
        result.success = (
            result.input_result.success and 
            all(pr.success for pr in result.processing_results) and
            result.reasoning_result.success
        )
        
        return result
    
    def process_benchmark(self, dataset_file_path: Union[str, Path], output_file_path: Union[str, Path], 
                         dataset_type: str = 'finqa', batch_size: Optional[int] = None) -> List[PipelineResult]:
        """Process benchmark dataset with multiple samples
        
        Args:
            dataset_file_path: Path to benchmark dataset file (JSON/JSONL)
            output_file_path: Path to save processed data
            dataset_type: Type of dataset ('finqa', 'tart', 'tablebench')
            batch_size: Number of samples to process in each batch (None for all)
            
        Returns:
            List of PipelineResult objects, one for each sample
        """
        dataset_file_path = Path(dataset_file_path)
        output_file_path = Path(output_file_path)
        
        # Update dataset uploader config
        self.dataset_uploader.config['dataset_type'] = dataset_type
        self.dataset_uploader.dataset_type = dataset_type
        self.dataset_uploader.field_mappings = self.dataset_uploader._get_field_mappings()
        
        # Update benchmark query processor config
        self.benchmark_query_processor.config['dataset_type'] = dataset_type
        self.benchmark_query_processor.dataset_type = dataset_type
        
        # Process dataset input
        dataset_result = self._process_dataset_input(dataset_file_path)
        
        if not dataset_result.success or dataset_result.data is None:
            logger.error(f"Dataset processing failed: {dataset_result.error_message}")
            raise Exception(f"Dataset processing failed: {dataset_result.error_message}")
        
        # Process each sample
        results = []
        samples_df = dataset_result.data
        
        # Apply batch processing if specified
        if batch_size:
            total_samples = len(samples_df)
            for i in range(0, total_samples, batch_size):
                batch_samples = samples_df.iloc[i:i+batch_size]
                batch_results = self._process_sample_batch(batch_samples, output_file_path, dataset_type)
                results.extend(batch_results)
        else:
            # Process all samples
            results = self._process_sample_batch(samples_df, output_file_path, dataset_type)
        
        return results
    
    def _process_sample_batch(self, samples_df: pd.DataFrame, output_file_path: Path, dataset_type: str) -> List[PipelineResult]:
        """Process a batch of samples"""
        results = []
        
        for idx, row in samples_df.iterrows():
            sample = row.to_dict()
            result = PipelineResult()
            
            try:
                # Extract table data
                table_data = pd.DataFrame(sample['table'])
                
                # Process query
                query_context = {
                    'table_info': {
                        'columns': sample['table_columns'],
                        'description': f"Table from {dataset_type} dataset"
                    },
                    **sample['context']
                }
                
                query_result = self._process_query(sample['query'], is_benchmark=True, context=query_context)
                
                if not query_result.metadata.validation_passed:
                    logger.warning(f"Query validation failed for sample {idx}")
                    result.success = False
                    result.error_message = "Query validation failed"
                    results.append(result)
                    continue
                
                # Process data (normalization)
                # Create a mock FileUploadeOutput for compatibility
                mock_input = type('MockInput', (), {
                    'data': table_data,
                    'success': True,
                    'error_message': None
                })()
                
                processing_results = self._process_data(mock_input, output_file_path)
                
                if not processing_results or not processing_results[-1].success:
                    error_msg = processing_results[-1].error_message if processing_results else "No processing results"
                    logger.error(f"Data processing failed for sample {idx}: {error_msg}")
                    result.success = False
                    result.error_message = error_msg
                    results.append(result)
                    continue
                
                # Process reasoning
                processed_data = processing_results[-1].data
                reasoning_result = self._process_reasoning_from_data(processed_data, query_result.data)
                
                # Set result data
                result.input_result = mock_input
                result.processing_results = processing_results
                result.reasoning_result = reasoning_result
                result.answer = reasoning_result.answer
                result.success = reasoning_result.success
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process sample {idx}: {str(e)}")
                result.success = False
                result.error_message = str(e)
                results.append(result)
        
        return results