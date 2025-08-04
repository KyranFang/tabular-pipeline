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

from tlp.input.base import BaseFileOperator, FileUploadeOutput, FileMetadata
from tlp.exceptions import FileFormatException, FileSizeException
from tlp.utils.utils import get_file_extension, detect_encoding, load_csv, load_excel, load_parquet, load_jsonl, decompress_file
from config.settings import settings

class FileUploader(BaseFileOperator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.max_file_size = self.config.get('max_file_size_mb', 100) * 1024 * 1024
        self.supported_formats = self.config.get('supported_formats', ['csv', 'xlsx', 'parquet', 'jsonl'])
        self.supported_compressions = self.config.get('supported_compressions', ['zip', 'gz', 'bz2'])
        
    def _load_data(self, file_path: Union[Path, str]) -> FileUploadeOutput:
        file_path = Path(file_path)
        if not self._validate_input(file_path):
            empty_metadata = FileMetadata(source_path=str(file_path))
            return FileUploadeOutput(
                data=None,
                metadata=empty_metadata,
                success=False,
                error_message="Invalid input file"
            )
        try:
            if file_path.suffix.lower() == '.zip':
                return self._load_zip_data(file_path)
            
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
            original_metadata = self._extract_metadata(file_path, data)
            
            others = {
                'id': original_metadata.id,
                'source_path': original_metadata.source_path,
                'file_size_bytes': original_metadata.file_size_bytes,
                'num_input_rows': original_metadata.num_input_rows,
                'num_input_columns': original_metadata.num_input_columns,
                'columns': original_metadata.columns
            }
            
            processed_sample = {
                'table': data,
                'query': None,
                'answer': None,
                'context': None,
                'others': others
            }
            
            processed_df = pd.DataFrame([processed_sample])
            metadata = FileMetadata(
                source_path=str(file_path),
                file_size_bytes=file_stats.st_size,
                num_input_rows=len(processed_df),
                num_input_columns=len(processed_df.columns),
                columns=processed_df.columns.tolist()
            )
            
            return FileUploadeOutput(
                data=processed_df,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            empty_metadata = FileMetadata(source_path=str(file_path))
            return FileUploadeOutput(
                data=None,
                metadata=empty_metadata,
                success=False,
                error_message=str(e)
            )
    
    def _load_zip_data(self, zip_path: Path) -> FileUploadeOutput:
        from config.settings import settings
        import tempfile
        import uuid
        
        temp_dir = settings.DATA_DIR / "temp" / str(uuid.uuid4())
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            processed_samples = []
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    if file_name.endswith('/'):
                        continue
                    
                    file_path = Path(file_name)
                    file_format = get_file_extension(file_path)
                    
                    if file_format not in self.supported_formats:
                        continue
                    
                    extracted_path = temp_dir / file_path.name
                    zip_ref.extract(file_name, temp_dir)
                    actual_extracted_path = temp_dir / file_name
                    
                    try:
                        encoding = detect_encoding(actual_extracted_path) if file_format in ['csv', 'tsv', 'xlsx', 'xls'] else None
                        
                        if file_format in ['csv', 'tsv']:
                            data = load_csv(actual_extracted_path, encoding)
                        elif file_format in ['xlsx', 'xls']:
                            data = load_excel(actual_extracted_path)
                        elif file_format == 'parquet':
                            data = load_parquet(actual_extracted_path)
                        elif file_format == 'jsonl':
                            data = load_jsonl(actual_extracted_path, encoding)
                        else:
                            continue
                        
                        original_metadata = self._extract_metadata(actual_extracted_path, data)
                        
                        others = {
                            'id': str(uuid.uuid4()),
                            'source_path': f"{zip_path}#{file_name}",
                            'file_size_bytes': actual_extracted_path.stat().st_size,
                            'num_input_rows': original_metadata.num_input_rows,
                            'num_input_columns': original_metadata.num_input_columns,
                            'columns': original_metadata.columns
                        }
                        
                        processed_sample = {
                            'table': data,
                            'query': None,
                            'answer': None,
                            'context': None,
                            'others': others
                        }
                        
                        processed_samples.append(processed_sample)
                        
                    except Exception:
                        continue
            
            if not processed_samples:
                raise FileFormatException("No valid table files found in ZIP")
            
            processed_df = pd.DataFrame(processed_samples)
            metadata = FileMetadata(
                source_path=str(zip_path),
                file_size_bytes=zip_path.stat().st_size,
                num_input_rows=len(processed_df),
                num_input_columns=len(processed_df.columns),
                columns=processed_df.columns.tolist()
            )
            
            return FileUploadeOutput(
                data=processed_df,
                metadata=metadata,
                success=True
            )
            
        finally:
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
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
    
    def process(self, source: Any, **kwargs) -> FileUploadeOutput:
        return self._load_data(source, **kwargs)

    
if __name__ == '__main__':
    uploader = FileUploader()
    sample_csv = Path("/home/fangnianrong/desktop/tabular-pipeline/data/sample/complex_sales_data.csv")
    result = uploader.process(sample_csv)
    
    # Test new meta-table format
    print("Meta-table structure:")
    print(f"Number of samples: {len(result.data)}")
    print(f"Columns: {result.data.columns.tolist()}")
    
    # Access the actual table data from first sample
    first_sample = result.data.iloc[0]
    actual_table = first_sample['table']
    others_info = first_sample['others']
    
    print("\nActual table data (first 2 rows):")
    print(actual_table.iloc[:2])
    print(f"\nSalesperson column dtype: {actual_table['Salesperson'].dtype}")
    print(f"\nOthers metadata: {others_info}")
    