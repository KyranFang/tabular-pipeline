import re
import numpy as np
import pandas as pd

from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from tlp.exceptions import NormalizationException
from tlp.processing.base import BaseProcessingOperator

class DataNormalizer(BaseProcessingOperator):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.normalize_column_names = self.config.get('normalize_column_names', True)
        self.normalize_data_types = self.config.get('normalize_data_types', True)
        self.normalize_missing_values = self.config.get('normalize_missing_values', True)
        self.remove_empty_rows = self.config.get('remove_empty_rows', True)
        self.remove_empty_columns = self.config.get('remove_empty_columns', True)
        self.imputation_strategy = self.config.get('imputation_strategy', 'none') # 'mean', 'median', 'mode', 'constant', 'none'
        
        self.auto_detect_types = self.config.get('auto_detect_types', True)
        self.date_formats = self.config.get('date_formats', [
            '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S'
        ])

        self.missing_value_representations = [
            'null', 'NULL', 'None', 'NONE', 'nan', 'NaN', 'NAN',
            'n/a', 'N/A', '#N/A', '#NULL!', '#DIV/0!', '-', '--', '?'
        ]
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        if data is None or len(data) == 0:
            self.logger.warning("Input data is empty")
            return False
        
        if len(data.columns) == 0:
            self.logger.warning("Input data has no columns")
            return False
        
        return True
    
    def _normalize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.normalize_column_names:
            return data
        
        new_columns = []
        for col in data.columns:
            col_str = str(col)
            col_str = col_str.strip()
            
            col_str = re.sub(r'[^a-zA-Z0-9_\u4e00-\u9fff]', '_', col_str)
            col_str = re.sub(r'_+', '_', col_str) # remove the continuous underscores
            col_str = col_str.strip('_')
            
            if not col or col_str == '_':
                default_col_str = f'default_column_{len(new_columns)}'
                self.logger.warning(f"Column name {col_str} is empty or only contains underscores, use the default name: {default_col_str}")
                col_str = default_col_str
            
            original_col = col_str
            counter = 1
            while col_str in new_columns:
                col_str = f"{original_col}_{counter}"
                counter += 1
                self.logger.warning(f"Column name {original_col} is repeated, use {col_str} instead")

            new_columns.append(col_str)
        
        data.columns = new_columns
        self.logger.debug(f"Normalized column names: {list(data.columns)}")
        return data
    
    def _normalize_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.normalize_missing_values:
            return data
        
        for col in data.columns:
            if data[col].dtype == 'object':
                # Replace strings consisting only of whitespace with NaN
                data[col] = data[col].replace(r'^\s*$', np.nan, regex=True)

        for missing_val in self.missing_value_representations:
            data = data.replace(missing_val, np.nan)
        
        for col in data.columns:
            if data[col].dtype == 'object':
                non_null_values = data[col].dropna()
                if len(non_null_values) > 0:
                    try:
                        pd.to_numeric(non_null_values, errors='raise')
                        data[col] = pd.to_numeric(data[col], errors='coerce')
                    except ValueError:
                        pass
                    
        # all nan-like cell has been rewritted to np.nan 
        # and all object-type column has been transformed to numeric column if available
        # begin to handle missing value heuristically
        if self.imputation_strategy == 'mean':
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    data[col] = data[col].fillna(data[col].mean())
        elif self.imputation_strategy == 'median':
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    data[col] = data[col].fillna(data[col].median())
        elif self.imputation_strategy == 'mode':
            for col in data.columns:
                data[col] = data[col].fillna(data[col].mode()[0])
        elif self.imputation_strategy == 'constant':
            for col in data.columns:
                if data[col].dtype in ['int64', 'float64']:
                    data[col] = data[col].fillna(0)
                elif data[col].dtype == 'object':
                    data[col] = data[col].fillna('Unknown')
        # 'none' strategy means no further imputation after initial NaN conversion

        return data
    
    def _detect_column_type(self, series: pd.Series) -> str:
        non_null_series = series.dropna()
        
        if len(non_null_series) == 0:
            return 'object'
        
        # Check data types in order of specificity
        type_checkers = [
            ('boolean', self._is_boolean_column),
            ('numeric', self._is_numeric_column),  
            ('datetime', self._is_datetime_column),
            ('category', self._is_categorical_column)
        ]
        
        for data_type, checker in type_checkers:
            if checker(non_null_series):
                if data_type == 'numeric':
                    return self._determine_numeric_subtype(non_null_series)
                return data_type
        
        return 'object'
    
    def _determine_numeric_subtype(self, series: pd.Series) -> str:
        """Determine if numeric data should be integer or float."""
        if self._is_integer_column(series):
            return 'integer'
        return 'float'
    
    # ============================================================================
    # Boolean Type Detection
    # ============================================================================
    
    def _is_boolean_column(self, series: pd.Series) -> bool:
        """Check if series contains boolean-like values."""
        unique_values = set(str(v).lower() for v in series.unique())
        boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 'y', 'n'}
        return unique_values.issubset(boolean_values) and len(unique_values) <= 2
    
    # ============================================================================
    # Numeric Type Detection
    # ============================================================================
    
    def _is_numeric_column(self, series: pd.Series) -> bool:
        """Check if series contains numeric values."""
        try:
            pd.to_numeric(series, errors='raise')
            return True
        except (ValueError, TypeError):
            return False
    
    def _is_integer_column(self, series: pd.Series) -> bool:
        """Check if numeric series contains only integer values."""
        try:
            numeric_series = pd.to_numeric(series, errors='raise')
            return (numeric_series % 1 == 0).all()
        except (ValueError, TypeError):
            return False
    
    # ============================================================================
    # DateTime Type Detection
    # ============================================================================
    
    def _is_datetime_column(self, series: pd.Series) -> bool:
        if self._try_predefined_datetime_formats(series):
            return True
        
        return self._try_automatic_datetime_inference(series)
    
    def _try_predefined_datetime_formats(self, series: pd.Series) -> bool:
        for date_format in self.date_formats:
            try:
                pd.to_datetime(series, format=date_format, errors='raise')
                return True
            except (ValueError, TypeError):
                continue
        return False
    
    def _try_automatic_datetime_inference(self, series: pd.Series) -> bool:
        try:
            pd.to_datetime(series, errors='raise', format='mixed')
            return True
        except (ValueError, TypeError):
            return False
    
    # ============================================================================
    # Categorical Type Detection
    # ============================================================================
    
    def _is_categorical_column(self, series: pd.Series) -> bool:
        """Check if series should be treated as categorical data."""
        if series.dtype != 'object':
            return False
        
        unique_count = series.nunique()
        total_count = len(series)
        
        # Consider categorical if unique values are less than 50 or 50% of total
        categorical_threshold = min(50, total_count * 0.5)
        return unique_count <= categorical_threshold
    
    def _convert_to_target_type(self, series: pd.Series, target_type: str) -> pd.Series:
        try:
            if target_type == 'boolean':
                bool_map = {
                    'true': True, 'false': False,
                    '1': True, '0': False,
                    'yes': True, 'no': False,
                    'y': True, 'n': False,
                }
                return series.astype(str).str.lower().map(bool_map).astype('boolean')
            
            elif target_type == 'integer':
                return pd.to_numeric(series, errors='coerce').astype('int64')
            
            elif target_type == 'float':
                return pd.to_numeric(series, errors='coerce').astype('float64')
            
            elif target_type == 'datetime':
                for date_format in self.date_formats:
                    try:
                        return pd.to_datetime(series, format=date_format, errors='raise')
                    except:
                        continue
                return pd.to_datetime(series, errors='coerce')
            
            elif target_type == 'category':
                return series.astype('category')
            
            else:  # object
                return series.astype('object')
                
        except Exception as e:
            self.logger.warning(f"Failed to convert column to {target_type}: {e}")
            return series
    
    def _normalize_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.normalize_data_types or not self.auto_detect_types:
            return data
        
        type_changes = {}
        
        for col in data.columns:
            original_type = str(data[col].dtype)
            detected_type = self._detect_column_type(data[col])
            
            if detected_type != 'object' or original_type == 'object':
                data[col] = self._convert_to_target_type(data[col], detected_type)
                new_type = str(data[col].dtype)
                
                if original_type != new_type:
                    type_changes[col] = {'from': original_type, 'to': new_type}
        
        if type_changes:
            self.logger.info(f"Data type changes: {type_changes}")
        
        return data
    
    def _remove_empty_rows_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        original_shape = data.shape
        
        if self.remove_empty_rows:
            data = data.dropna(how='all')
        if self.remove_empty_columns:
            data = data.dropna(axis=1, how='all')
        
        new_shape = data.shape
        if original_shape != new_shape:
            self.logger.info(f"Removed empty rows/columns: {original_shape} -> {new_shape}")
        
        return data
    
    def _transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._normalize_column_names(data)
        data = self._normalize_missing_values(data)
        data = self._remove_empty_rows_columns(data)
        data = self._normalize_data_types(data)
        
        return data
