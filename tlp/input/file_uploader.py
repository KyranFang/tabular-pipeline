"""File upload processing module"""
import os
import bz2
import gzip
import zipfile
import chardet
import pandas as pd

from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Set

from tlp.input.base import BaseFileOperator, InputResult, FileMetadata
from tlp.exceptions import FileFormatException, FileSizeException
from tlp.utils.utils import get_file_extension, detect_encoding, load_csv, load_excel, load_parquet, load_jsonl, decompress_file
from config.settings import settings

class FileUploader(BaseFileOperator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_file_size = self.config.get('max_file_size_mb', 100) * 1024 * 1024
        self.supported_formats = self.config.get('supported_formats', ['csv', 'xlsx', 'parquet', 'jsonl'])
        self.supported_compressions = self.config.get('supported_compressions', ['zip', 'gz', 'bz2'])
        
    def _load_data(self, file_path: Union[Path, str]) -> InputResult:
        file_path = Path(file_path)
        if not self._validate_input(file_path):
            # Create empty metadata for failed validation
            empty_metadata = FileMetadata(source_path=str(file_path))
            return InputResult(
                data=None,
                metadata=empty_metadata,
                success=False,
                error_message="Invalid input file"
            )
        try:
            actual_file_path = decompress_file(file_path)
            file_format = get_file_extension(actual_file_path)
            encoding = detect_encoding(actual_file_path) if file_format in ['csv', 'tsv', 'xlsx', 'xls'] else None
            
            if file_format in ['csv', 'tsv']:
                data = load_csv(actual_file_path, encoding)
            elif file_format in ['xlsx', 'xls']:
                data = load_excel(actual_file_path)
            elif file_format == 'parquet':
                data = load_parquet(actual_file_path)
            elif file_format == 'jsonl':
                data = load_jsonl(actual_file_path, encoding)
            else:
                raise FileFormatException(f"Unsupported file format: {file_format}")
            
            file_stats = file_path.stat()
            metadata = self._extract_metadata(file_path, data)
            
            return InputResult(
                data=data,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            empty_metadata = FileMetadata(source_path=str(file_path))
            return InputResult(
                data=None,
                metadata=empty_metadata,
                success=False,
                error_message=str(e)
            )
    
    def _validate_input(self, file_path: Union[Path, str]) -> bool:
        if isinstance(file_path, (str, Path)):
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return False
            
            if file_path.stat().st_size > self.max_file_size:
                self.logger.error(f"File size exceeds maximum limit: {file_path}")
                return False

            file_extension = get_file_extension(file_path)
            if file_extension not in self.supported_compressions + self.supported_formats:
                self.logger.error(f"Unsupported file format: {file_extension}")
                return False
            
            return True
        return False
    
    def _save_data(self, result: Any, file_path: Union[Path, str]) -> bool:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json_data = {
                    "id": getattr(result, 'id', str(uuid.uuid4())),
                    "data": result.data.to_json(orient='records') if hasattr(result.data, 'to_json') else str(result.data),
                    "metadata": result.metadata.dict() if hasattr(result.metadata, 'dict') else result.metadata,
                    "success": getattr(result, 'success', True),
                    "error_message": getattr(result, 'error_message', None)
                }
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')
            return True
        except Exception as e:
            self.logger.error(f"Failed to save data: {e}")
            return False
    
    def process(self, source: Any, **kwargs) -> InputResult:
        return self._load_data(source, **kwargs)

    
if __name__ == '__main__':
    uploader = FileUploader()
    sample_csv = Path("/home/fangnianrong/desktop/tabular-pipeline/data/sample/complex_sales_data.csv")
    result = uploader.process(sample_csv)
    print(result.data.iloc[:2])
    print(result.data['Salesperson'].dtype)
    uploader.save_data(result, output_dir=Path("./test_output"), file_prefix="test_serialization")
    