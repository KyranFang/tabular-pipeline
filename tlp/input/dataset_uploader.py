import io
import json
import pandas as pd

from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Set

from tlp.utils.logger import get_logger
from tlp.input import BaseFileOperator, FileUploadeOutput, FileMetadata


logger = get_logger(__name__)


class DatasetUploader(BaseFileOperator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.dataset_name = config.get('dataset_name', '')
        self.dataset_path = config.get('dataset_path', '')
        self.dataset_feature = config.get('dataset_feature', {'table':None, 
                                                              'query':None, 
                                                              'answer':None, 
                                                              'context':None})

    def _get_feature_mapping(self, real_dataset_feature: List[str]) -> None:
        required: Dict[str, str] = {
            'table': self.dataset_feature['table'],
            'query': self.dataset_feature['query'],
        }

        optional: Dict[str, str] = {
            'answer': self.dataset_feature.get('answer'),
            'context': self.dataset_feature.get('context'),
        }

        all_mapped: Set[str] = set(required.values()) | {v for v in optional.values() if v}

        missing = [f"{k} field '{v}'" for k, v in required.items()
                   if v not in real_dataset_feature]
        if missing:
            raise ValueError(f"Missing required fields in dataset: {', '.join(missing)}")

        self.others_features = [f for f in real_dataset_feature if f not in all_mapped]

        self.table_feature   = required['table']
        self.query_feature   = required['query']
        self.answer_feature  = optional['answer']
        self.context_feature = optional['context']

    def _load_data(self) -> (pd.DataFrame, List[str]):
        dataset_path = Path(self.dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"File not found: {dataset_path}")
        assert dataset_path.suffix in [".json", ".jsonl"], ValueError(f"Unsupported file format: {dataset_path.suffix}, please use json or jsonl")
        
        dataset = pd.read_json(dataset_path)
        dataset_feature = dataset.columns
        
        return dataset, dataset_feature
    
    def _save_data(self):
        raise NotImplementedError("Save data method do not need to be implemented in this case")

    def _extract_table_data(self, sample: Dict[str, Any]) -> pd.DataFrame:
        if not self.table_feature or self.table_feature not in sample:
            raise ValueError(f"Table field '{self.table_feature}' not found in sample")
            
        table_data = sample[self.table_feature]
        
        if isinstance(table_data, str):
            return pd.read_json(io.StringIO(table_data))
        elif isinstance(table_data, list) or isinstance(table_data, dict):
            return pd.DataFrame(table_data)
        else:
            raise ValueError(f"Unsupported table data format: {type(table_data)}")
    
    def _extract_field_data(self, sample: Dict[str, Any], field_name: str) -> Any:
        if field_name and field_name in sample:
            return sample[field_name]
        return None
    
    def _extract_others(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        others = {}
        for field in self.others_features:
            if field in sample:
                others[field] = sample[field]
        return others

    def process(self) -> FileUploadeOutput:
        dataset, dataset_feature = self._load_data()
        self._get_feature_mapping(dataset_feature)
        
        processed_samples = []
        failed_cnt = 0
        for _, row in dataset.iterrows():
            sample = row.to_dict()
            
            try:
                table_df = self._extract_table_data(sample)
                query = self._extract_field_data(sample, self.query_feature)
                answer = self._extract_field_data(sample, self.answer_feature)
                context = self._extract_field_data(sample, self.context_feature)
                others = self._extract_others(sample)
                
                processed_sample = {
                    'table': table_df,
                    'query': query,
                    'answer': answer,
                    'context': context,
                    'others': others
                }
                
                processed_samples.append(processed_sample)
                
            except Exception as e:
                failed_cnt += 1
                logger.warning(f"Failed to process sample: {e}")
                if failed_cnt >= 10:
                    raise TLPError(f"Failed to process 10 samples, stop uploading, please check the dataset")
                continue
        
        processed_df = pd.DataFrame(processed_samples)
        
        # Get file size for metadata
        dataset_path = Path(self.dataset_path)
        file_size = dataset_path.stat().st_size if dataset_path.exists() else 0
        
        metadata = FileMetadata(
            source_path=self.dataset_path,
            file_size_bytes=file_size,
            num_input_rows=len(processed_df),
            num_input_columns=len(processed_df.columns),
            columns=processed_df.columns.tolist()
        )
        
        return FileUploadeOutput(
            data=processed_df,
            metadata=metadata,
            success=True
        )
