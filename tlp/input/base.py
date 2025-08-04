import json
import uuid
import pandas as pd

from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

from tlp.utils.logger import get_logger
from tlp.data_structure import BaseModuleOutput, BaseMetadata
from tlp.exceptions import InputException, FileFormatException
from tlp.utils.utils import get_file_extension, detect_encoding, load_csv, load_excel, load_parquet, load_jsonl, decompress_file


logger = get_logger(__name__)

class FileMetadata(BaseMetadata):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_path: Optional[str] = None
    file_size_bytes: int = 0
    num_input_rows: int = 0
    num_input_columns: int = 0
    columns: Optional[List[str]] = []
    
class FileUploadeOutput(BaseModuleOutput):
    # Inherits all fields from BaseModuleOutput
    # Override metadata type to be more specific
    metadata: FileMetadata = Field(alias='_metadata')
    
    class Config:
        populate_by_name = True
        validate_by_name = True
    
class BaseFileOperator(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
    
    @abstractmethod
    def _load_data(self, source: Any, **kwargs) -> FileUploadeOutput:
        pass

    @abstractmethod
    def _save_data(self, result: Any, file_path: Union[Path, str]) -> bool:
        pass

    @abstractmethod
    def process(self, source: Any, **kwargs) -> FileUploadeOutput:
        pass

    def _extract_metadata(self, file_path: Path, data:pd.DataFrame) -> FileMetadata:
        return FileMetadata(
            source_path=str(file_path),
            file_size_bytes=file_path.stat().st_size if file_path.exists() else None,
            num_input_rows=len(data),
            num_input_columns=len(data.columns),
            columns=data.columns.tolist()
        )
    