#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Table Corpus Management Layer

Provides lightweight CRUD operations for table corpus with snapshot mechanism.
Maintains compatibility with existing FileUploader/DatasetUploader meta-table format.
"""

import uuid
import pandas as pd
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
from datetime import datetime
from copy import deepcopy

from tlp.utils.logger import get_logger
from tlp.input.base import FileMetadata, FileUploadeOutput
from tlp.input.file_uploader import FileUploader
from tlp.input.dataset_uploader import DatasetUploader


logger = get_logger(__name__)


class TableCorpusSnapshot:
    """
    Immutable snapshot of table corpus state.
    Maintains meta-table format compatibility with DatasetUploader.
    """
    
    def __init__(self, data: pd.DataFrame, metadata: FileMetadata):
        self.data = data.copy()  # Immutable copy
        self.metadata = metadata
        self.timestamp = datetime.now()
    
    def get_table_by_id(self, table_id: str) -> Optional[Dict[str, Any]]:
        """Get table record by unique ID"""
        matches = self.data[self.data['others'].apply(lambda x: x.get('id') == table_id)]
        if len(matches) > 0:
            return matches.iloc[0].to_dict()
        return None
    
    def get_tables_by_source(self, source_path: str) -> List[Dict[str, Any]]:
        """Get table records by source path"""
        matches = self.data[self.data['others'].apply(lambda x: x.get('source_path') == source_path)]
        return [row.to_dict() for _, row in matches.iterrows()]
    
    def list_all_ids(self) -> List[str]:
        """Get all table IDs in corpus"""
        return [others.get('id') for others in self.data['others']]
    
    def get_corpus_stats(self) -> Dict[str, Any]:
        """Get corpus statistics"""
        return {
            'total_tables': len(self.data),
            'unique_sources': len(set(others.get('source_path') for others in self.data['others'])),
            'timestamp': self.timestamp,
            'total_rows': sum(others.get('num_input_rows', 0) for others in self.data['others']),
            'total_columns': sum(others.get('num_input_columns', 0) for others in self.data['others'])
        }


class TableCorpus:
    """
    Lightweight table corpus management with CRUD operations.
    Maintains compatibility with FileUploader/DatasetUploader meta-table format.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger
        
        # Internal storage: DataFrame with meta-table format
        self._corpus_data = pd.DataFrame(columns=['table', 'query', 'answer', 'context', 'others'])
        
        # Index for fast lookup
        self._id_index: Dict[str, int] = {}  # id -> row_index
        self._source_index: Dict[str, List[int]] = {}  # source_path -> [row_indices]
        
        # File uploaders for adding new tables
        self.file_uploader = FileUploader(config)
        
    def add_from_file(self, file_path: Union[Path, str], custom_id: Optional[str] = None) -> Union[str, List[str]]:
        """
        Add table(s) from file using FileUploader.
        Supports both single files and ZIP files containing multiple tables.
        
        Args:
            file_path: Path to file to upload
            custom_id: Optional custom ID for single table, ignored for ZIP files
            
        Returns:
            str: ID of added table (for single files)
            List[str]: List of IDs of added tables (for ZIP files)
        """
        result = self.file_uploader.process(file_path)
        if not result.success:
            raise ValueError(f"Failed to upload file {file_path}: {result.error_message}")
        
        # Handle multiple tables (ZIP files)
        if len(result.data) > 1:
            added_ids = []
            for idx in range(len(result.data)):
                table_record = result.data.iloc[idx].to_dict()
                table_id = self._add_table_record(table_record)
                added_ids.append(table_id)
            return added_ids
        
        # Handle single table
        uploaded_table = result.data.iloc[0].to_dict()
        
        # Override ID if custom_id provided
        if custom_id:
            uploaded_table['others']['id'] = custom_id
        
        return self._add_table_record(uploaded_table)
    
    def add_from_dataset(self, dataset_uploader: DatasetUploader, table_indices: Optional[List[int]] = None) -> List[str]:
        """
        Add table(s) from DatasetUploader result.
        
        Args:
            dataset_uploader: Configured DatasetUploader instance
            table_indices: Optional list of indices to add, otherwise add all
            
        Returns:
            List[str]: List of IDs of added tables
        """
        result = dataset_uploader.process()
        if not result.success:
            raise ValueError(f"Failed to process dataset: {result.error_message}")
        
        added_ids = []
        indices_to_add = table_indices if table_indices is not None else range(len(result.data))
        
        for idx in indices_to_add:
            if idx >= len(result.data):
                logger.warning(f"Index {idx} out of range, skipping")
                continue
                
            table_record = result.data.iloc[idx].to_dict()
            table_id = self._add_table_record(table_record)
            added_ids.append(table_id)
        
        return added_ids
    
    def add_table_direct(self, table: pd.DataFrame, others: Dict[str, Any], 
                        query: Any = None, answer: Any = None, context: Any = None) -> str:
        """
        Add table directly with metadata.
        
        Args:
            table: pandas DataFrame
            others: Metadata dictionary (must contain 'id' or will be auto-generated)
            query: Optional query data
            answer: Optional answer data
            context: Optional context data
            
        Returns:
            str: ID of added table
        """
        # Ensure ID exists
        if 'id' not in others:
            others['id'] = str(uuid.uuid4())
        
        table_record = {
            'table': table,
            'query': query,
            'answer': answer,
            'context': context,
            'others': others
        }
        
        return self._add_table_record(table_record)
    
    def delete_by_id(self, table_id: str) -> bool:
        """
        Delete table by ID.
        
        Args:
            table_id: Unique table ID
            
        Returns:
            bool: True if deleted, False if not found
        """
        if table_id not in self._id_index:
            return False
        
        row_idx = self._id_index[table_id]
        source_path = self._corpus_data.iloc[row_idx]['others'].get('source_path')
        
        # Remove from DataFrame
        self._corpus_data = self._corpus_data.drop(self._corpus_data.index[row_idx]).reset_index(drop=True)
        
        # Update indices
        self._rebuild_indices()
        
        logger.info(f"Deleted table {table_id} from corpus")
        return True
    
    def delete_by_source(self, source_path: str) -> int:
        """
        Delete all tables from specific source.
        
        Args:
            source_path: Source file path
            
        Returns:
            int: Number of tables deleted
        """
        if source_path not in self._source_index:
            return 0
        
        # Get all IDs for this source
        row_indices = self._source_index[source_path]
        table_ids = [self._corpus_data.iloc[idx]['others']['id'] for idx in row_indices]
        
        # Delete each table
        deleted_count = 0
        for table_id in table_ids:
            if self.delete_by_id(table_id):
                deleted_count += 1
        
        return deleted_count
    
    def update_table(self, table_id: str, table: Optional[pd.DataFrame] = None, 
                    others: Optional[Dict[str, Any]] = None,
                    query: Any = None, answer: Any = None, context: Any = None) -> bool:
        """
        Update table and/or metadata by ID.
        
        Args:
            table_id: Unique table ID
            table: Optional new table data
            others: Optional new metadata (will be merged with existing)
            query: Optional new query data
            answer: Optional new answer data
            context: Optional new context data
            
        Returns:
            bool: True if updated, False if not found
        """
        if table_id not in self._id_index:
            return False
        
        row_idx = self._id_index[table_id]
        
        # Update fields if provided
        if table is not None:
            self._corpus_data.at[row_idx, 'table'] = table
        
        if others is not None:
            # Merge with existing others, preserving ID
            current_others = self._corpus_data.iloc[row_idx]['others'].copy()
            current_others.update(others)
            current_others['id'] = table_id  # Preserve ID
            self._corpus_data.at[row_idx, 'others'] = current_others
        
        if query is not None:
            self._corpus_data.at[row_idx, 'query'] = query
        
        if answer is not None:
            self._corpus_data.at[row_idx, 'answer'] = answer
        
        if context is not None:
            self._corpus_data.at[row_idx, 'context'] = context
        
        # Rebuild indices if source_path changed
        if others and 'source_path' in others:
            self._rebuild_indices()
        
        logger.info(f"Updated table {table_id} in corpus")
        return True
    
    def get_snapshot(self) -> TableCorpusSnapshot:
        """
        Get immutable snapshot of current corpus state.
        
        Returns:
            TableCorpusSnapshot: Immutable corpus snapshot
        """
        metadata = FileMetadata(
            source_path="table_corpus",
            file_size_bytes=0,  # Not applicable for corpus
            num_input_rows=len(self._corpus_data),
            num_input_columns=len(self._corpus_data.columns) if len(self._corpus_data) > 0 else 5,
            columns=self._corpus_data.columns.tolist()
        )
        
        return TableCorpusSnapshot(self._corpus_data, metadata)
    
    def clear(self) -> None:
        """Clear all tables from corpus"""
        self._corpus_data = pd.DataFrame(columns=['table', 'query', 'answer', 'context', 'others'])
        self._id_index.clear()
        self._source_index.clear()
        logger.info("Cleared table corpus")
    
    def save_to_file(self, file_path: Union[Path, str]) -> bool:
        """
        Save corpus to file for persistent storage.
        
        Args:
            file_path: Path to save the corpus file
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            corpus_state = {
                'corpus_data': self._corpus_data,
                'config': self.config,
                'timestamp': datetime.now(),
                'version': '1.0'
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(corpus_state, f)
            
            logger.info(f"Saved corpus to {file_path} with {len(self._corpus_data)} tables")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save corpus to {file_path}: {str(e)}")
            return False
    
    def load_from_file(self, file_path: Union[Path, str]) -> bool:
        """
        Load corpus from file.
        
        Args:
            file_path: Path to the corpus file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"Corpus file {file_path} does not exist")
                return False
            
            with open(file_path, 'rb') as f:
                corpus_state = pickle.load(f)
            
            # Validate loaded data
            if not isinstance(corpus_state, dict) or 'corpus_data' not in corpus_state:
                logger.error(f"Invalid corpus file format: {file_path}")
                return False
            
            # Restore corpus state
            self._corpus_data = corpus_state['corpus_data']
            if 'config' in corpus_state:
                self.config.update(corpus_state.get('config', {}))
            
            # Rebuild indices
            self._rebuild_indices()
            
            logger.info(f"Loaded corpus from {file_path} with {len(self._corpus_data)} tables")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load corpus from {file_path}: {str(e)}")
            return False
    
    @classmethod
    def from_file(cls, file_path: Union[Path, str], config: Optional[Dict[str, Any]] = None) -> Optional['TableCorpus']:
        """
        Create TableCorpus instance from saved file.
        
        Args:
            file_path: Path to the corpus file
            config: Optional config to override saved config
            
        Returns:
            TableCorpus: Loaded corpus instance, or None if failed
        """
        corpus = cls(config)
        if corpus.load_from_file(file_path):
            return corpus
        return None
    
    def _add_table_record(self, table_record: Dict[str, Any]) -> str:
        """Internal method to add table record to corpus"""
        table_id = table_record['others']['id']
        
        # Check for duplicate ID
        if table_id in self._id_index:
            raise ValueError(f"Table with ID {table_id} already exists")
        
        # Add to DataFrame
        new_row = pd.DataFrame([table_record])
        self._corpus_data = pd.concat([self._corpus_data, new_row], ignore_index=True)
        
        # Update indices
        row_idx = len(self._corpus_data) - 1
        self._id_index[table_id] = row_idx
        
        source_path = table_record['others'].get('source_path')
        if source_path:
            if source_path not in self._source_index:
                self._source_index[source_path] = []
            self._source_index[source_path].append(row_idx)
        
        logger.info(f"Added table {table_id} to corpus")
        return table_id
    
    def _rebuild_indices(self) -> None:
        """Rebuild internal indices after DataFrame modifications"""
        self._id_index.clear()
        self._source_index.clear()
        
        for idx, row in self._corpus_data.iterrows():
            table_id = row['others']['id']
            self._id_index[table_id] = idx
            
            source_path = row['others'].get('source_path')
            if source_path:
                if source_path not in self._source_index:
                    self._source_index[source_path] = []
                self._source_index[source_path].append(idx)


if __name__ == '__main__':
    # Example usage
    corpus = TableCorpus()
    
    # Add table from file
    sample_csv = Path("/home/fangnianrong/desktop/tabular-pipeline/data/sample/complex_sales_data.csv")
    table_id = corpus.add_from_file(sample_csv)
    print(f"Added table with ID: {table_id}")
    
    # Get snapshot
    snapshot = corpus.get_snapshot()
    print(f"Corpus stats: {snapshot.get_corpus_stats()}")
    
    # Test retrieval
    table_record = snapshot.get_table_by_id(table_id)
    if table_record:
        print(f"Retrieved table shape: {table_record['table'].shape}")
        print(f"Table metadata: {table_record['others']}")