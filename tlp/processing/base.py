import uuid
import json
import pandas as pd


from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

from tlp.utils.logger import get_logger
from tlp.exceptions import ProcessingException

logger = get_logger(__name__)


class ProcessingMetadata(BaseModel):
    step_name: str
    num_output_rows: int
    num_output_columns: int
    processing_time_seconds: float
    columns: Optional[List[str]] = None
    sample_rows: Optional[List[Dict[str, Any]]] = None
    warnings: List[str] = []
    source_path: Optional[str] = None

class ProcessingResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data: Any  # pandas.DataFrame
    metadata: ProcessingMetadata
    success: bool = True
    error_message: Optional[str] = None

class BaseProcessingOperator(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        self.step_name = self.__class__.__name__
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        pass
    
    @abstractmethod
    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def validate_result(self, input_data: pd.DataFrame, output_data: pd.DataFrame) -> List[str]:
        warnings = []
        
        if output_data is None or len(output_data) == 0:
            warnings.append("Output data is empty")
        
        if len(output_data.columns) == 0:
            warnings.append("Output data has no columns")
        
        row_change_ratio = (len(output_data) - len(input_data)) / len(input_data) if len(input_data) > 0 else 0
        if abs(row_change_ratio) > 0.5:
            warnings.append(f"Significant row count change: {row_change_ratio:.2%}")
        
        col_change_ratio = (len(output_data.columns) - len(input_data.columns)) / len(input_data.columns) if len(input_data.columns) > 0 else 0
        if abs(col_change_ratio) > 0.3:
            warnings.append(f"Significant column count change: {col_change_ratio:.2%}")
        
        return warnings
    
    def save_result(self, result: ProcessingResult, output_path: Union[str, Path]) -> bool:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json_data = {
                    "id": result.id,
                    "data": result.data.to_json(orient='records', date_format='iso') if hasattr(result.data, 'to_json') else str(result.data),
                    "metadata": result.metadata.model_dump(),
                    "success": result.success,
                    "error_message": result.error_message
                }
                f.write(json.dumps(json_data, ensure_ascii=False, default=str) + '\n')
            return True
        except Exception as e:
            self.logger.error(f"Failed to save result: {e}")
            return False
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        idx = None
        source_path = None
        
        if isinstance(data, pd.DataFrame):
            data = data
            metadata = None
        else:
            assert hasattr(data, 'data')
            actual_data = getattr(data, 'data')
            metadata = getattr(data, 'metadata', None)
            idx = getattr(data, 'id', None) if hasattr(data, 'id') else None
            source_path = getattr(metadata, 'source_path', None) if metadata and hasattr(metadata, 'source_path') else None
            data = actual_data
        
        start_time = datetime.now()
        
        try:
            self.logger.log_step(f"{self.step_name}.process", "START")
            
            if not self.validate_data(data):
                raise ProcessingException(f"Invalid input data for {self.step_name}")
            
            input_rows, input_columns = len(data), len(data.columns)
            transformed_data = self._transform(data)
            if transformed_data is None:
                raise ProcessingException(f"Transform returned None for {self.step_name}")
            
            output_rows, output_columns = len(transformed_data), len(transformed_data.columns)
            
            warnings = self.validate_result(data, transformed_data)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            metadata = ProcessingMetadata(
                step_name=self.step_name,
                num_output_rows=output_rows,
                num_output_columns=output_columns,
                processing_time_seconds=processing_time,
                columns=transformed_data.columns.tolist(),
                sample_rows=transformed_data.head().to_dict(orient='records'),
                warnings=warnings,
                source_path=source_path
            )
            
            self.logger.log_step(
                f"{self.step_name}.process",
                "SUCCESS",
                input_shape=(input_rows, input_columns),
                output_shape=(output_rows, output_columns),
                processing_time=processing_time,
                warnings_count=len(warnings)
            )
            
            if warnings:
                for warning in warnings:
                    self.logger.warning(f"{self.step_name}: {warning}")
            
            result_kwargs = {
                'data': transformed_data,
                'metadata': metadata,
                'success': True
            }
            if idx is not None:
                result_kwargs['id'] = idx
            return ProcessingResult(**result_kwargs)
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"{self.step_name} processing failed: {str(e)}")
            metadata=ProcessingMetadata(
                    step_name=self.step_name,
                    num_output_columns=len(data.columns),
                    num_output_rows=len(data),
                    columns=data.columns.tolist(),
                    sample_rows=data.head().to_dict(orient='records'),
                    processing_time_seconds=processing_time,
                    warnings=[str(e)],
                    source_path=source_path
                )
            result_kwargs = {
                'data': data,
                'metadata': metadata,
                'success': False,
                'error_message': str(e)
            }
            if idx is not None:
                result_kwargs['id'] = idx
            return ProcessingResult(**result_kwargs)
    
    def is_enabled(self) -> bool:
        return self.config.get('enabled', True)
    
    def should_skip(self) -> bool:
        return False if self.is_enabled() else True


class ConditionalProcessingOperator(BaseProcessingOperator):
    @abstractmethod
    def check_condition(self, data: pd.DataFrame) -> bool:
        pass
    
    def should_skip(self, data: pd.DataFrame) -> bool:
        """Override skip logic"""
        if not self.is_enabled():
            return True
        
        if not self.check_condition(data):
            self.logger.info(f"{self.step_name}: Condition not met, skipping")
            return True
        
        return False


class ChainableProcessingOperator(BaseProcessingOperator):
    """Chainable processor base class"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.next_operator: Optional[BaseProcessingOperator] = None
    
    def set_next(self, operator: BaseProcessingOperator) -> BaseProcessingOperator:
        """Set next processor"""
        self.next_operator = operator
        return operator
    
    def process_chain(self, data: pd.DataFrame, **kwargs) -> List[ProcessingResult]:
        """Process entire chain"""
        results = []
        current_data = data
        current_operator = self
        
        while current_operator is not None:
            if current_operator.should_skip(current_data):
                self.logger.info(f"Skipping {current_operator.step_name}")
                current_operator = current_operator.next_operator
                continue
            
            result = current_operator.process(current_data, **kwargs)
            results.append(result)
            
            if not result.success:
                self.logger.error(f"Chain processing stopped at {current_operator.step_name}")
                break
            
            current_data = result.data
            current_operator = current_operator.next_operator
        
        return results