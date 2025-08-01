import json
import uuid
import pandas as pd

from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

from tlp.utils.logger import get_logger
from tlp.exceptions import InputException, FileFormatException
from tlp.utils.utils import get_file_extension, detect_encoding, load_csv, load_excel, load_parquet, load_jsonl, decompress_file


logger = get_logger(__name__)

class FileMetadata(BaseModel):
    source_path: Optional[str] = None
    file_size_bytes: int = 0
    num_input_rows: int = 0
    num_input_columns: int = 0
    columns: Optional[List[str]] = []
    
class InputResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    data: Any # normally pandas.Dataframe
    metadata: FileMetadata
    success: bool = True
    error_message: Optional[str] = None
    
class BaseFileOperator(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
    
    @abstractmethod
    def _load_data(self, source: Any, **kwargs) -> InputResult:
        pass

    @abstractmethod
    def _save_data(self, result: Any, file_path: Union[Path, str]) -> bool:
        pass

    @abstractmethod
    def process(self, source: Any, **kwargs) -> InputResult:
        pass

    def _extract_metadata(self, file_path: Path, data:pd.DataFrame) -> FileMetadata:
        return FileMetadata(
            source_path=str(file_path),
            file_size_bytes=file_path.stat().st_size if file_path.exists() else None,
            num_input_rows=len(data),
            num_input_columns=len(data.columns),
            columns=data.columns.tolist()
        )
    