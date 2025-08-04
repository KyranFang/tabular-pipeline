# Tabular Pipeline (TLP) - Developer Documentation

**Version:** 0.1.1

## What's new

- **0.1.1**: 
  - Added support for multiple table uploads in ZIP files in FileUploader
  - Enhanced TableCorpus to handle batch file uploads from ZIP archives
  - Improved compatibility between single file and batch processing workflows
  - Maintained backward compatibility for existing single file upload functionality
- **0.1.0**: Initial release with core components and basic functionality

## Overview

Tabular Pipeline (TLP) is a modular data processing and reasoning framework designed for tabular data analysis. The system provides a complete pipeline from data ingestion to intelligent reasoning, with support for multiple file formats and flexible processing workflows.

## Architecture

The TLP framework follows a modular architecture with four main components:

```
Input → Processing → Reasoning → Output
```

### Core Components

#### 1. Data Structures (`tlp/data_structure/`)

The foundation of the system is built on standardized data structures:

- **`BaseMetadata`**: Base class for metadata storage
- **`BaseModuleOutput`**: Standardized output format for all modules
  - `id`: Unique identifier for each operation
  - `data`: The actual data payload
  - `metadata`: Module-specific metadata
  - `success`: Operation success flag
  - `error_message`: Error details if operation fails

#### 2. Input Module (`tlp/input/`)

Handles data ingestion from various file formats:

- **`BaseFileOperator`**: Abstract base class for file operations
- **`FileUploader`**: Concrete implementation supporting:
  - CSV, TSV files
  - Excel files (XLSX, XLS)
  - Parquet files
  - JSONL files
  - Compressed files (ZIP, GZ, BZ2)

**Key Features:**
- Automatic encoding detection
- File size validation
- Format validation
- Metadata extraction (file size, row/column counts, column names)

**Output Structure:**
```python
class FileUploadeOutput(BaseModuleOutput):
    metadata: FileMetadata  # Contains source_path, file_size_bytes, num_input_rows, etc.
```

#### 3. Processing Module (`tlp/processing/`)

Transforms and normalizes data:

- **`BaseProcessingOperator`**: Abstract base for processing operations
- **`DataNormalizer`**: Handles data normalization:
  - Column name standardization
  - Data type detection and conversion
  - Missing value handling
  - Empty row/column removal
  - Data imputation strategies

**Processing Chain Support:**
- **`ConditionalProcessingOperator`**: Conditional processing based on data characteristics
- **`ChainableProcessingOperator`**: Supports chaining multiple processing steps

**Output Structure:**
```python
class ProcessingOutput(BaseModuleOutput):
    metadata: ProcessingMetadata  # Contains step_name, processing_time, warnings, etc.
```

#### 4. Reasoning Module (`tlp/reasoning/`)

Provides intelligent analysis capabilities:

- **`BaseReasoner`**: Abstract base for reasoning operations
- **`SimpleReasoner`**: Basic reasoning implementation with LLM integration
- **`BaseQueryProcessor`**: Query analysis and decomposition
- **`SimpleQueryProcessor`**: Basic query processing

**Query Types:**
- `SIMPLE`: Basic queries
- `STATISTICAL`: Statistical analysis
- `FILTER`: Data filtering
- `PREDICTION`: Predictive analysis
- `GENERATION`: Data generation
- `COMPARISON`: Comparative analysis
- `ANOMALY`: Anomaly detection
- `COMPLEX`: Multi-step complex queries

**Reasoning Paths:**
- `DIRECT_REASONING`: Direct LLM-based reasoning
- `SQL_TOOL`: SQL-based analysis
- `PYTHON_TOOL`: Python code execution
- `HYBRID`: Combined approaches

**Output Structure:**
```python
class ReasoningOutput(BaseModuleOutput):
    metadata: ReasoningMetadata  # Contains reasoning_path, prompts, etc.
    
    @property
    def answer(self) -> str:  # The reasoning result
```

## Data Flow

The system processes data through the following stages:

### 1. Input Stage
```
File Input → FileUploader → FileUploadeOutput
```
- Validates file format and size
- Detects encoding and loads data into pandas DataFrame
- Extracts metadata (file info, dimensions, column names)
- Returns standardized output with success/error status

### 2. Processing Stage
```
FileUploadeOutput → DataNormalizer → ProcessingOutput
```
- Normalizes column names (lowercase, underscore separation)
- Detects and converts data types automatically
- Handles missing values based on configured strategy
- Removes empty rows/columns if configured
- Saves processed data to JSONL format

### 3. Reasoning Stage
```
ProcessingOutput/File → SimpleReasoner → ReasoningOutput
```
- Processes user queries through QueryProcessor
- Supports two modes:
  - **Direct reasoning**: Uses processed DataFrame directly
  - **File-based reasoning**: Reads from saved JSONL file
- Integrates with local LLM for intelligent analysis
- Returns structured reasoning results

## Pipeline Orchestration

The `Pipeline` class orchestrates the entire workflow:

```python
class Pipeline:
    def process(self, input_file_path, output_file_path, query, use_saved_data=False):
        # 1. Input processing
        input_result = self._process_input(input_file_path)
        
        # 2. Data processing
        processing_results = self._process_data(input_result.data, output_file_path)
        
        # 3. Reasoning (two modes)
        if use_saved_data:
            reasoning_result = self._process_reasoning_from_file(output_file_path, query)
        else:
            reasoning_result = self._process_reasoning_from_data(processed_data, query)
        
        # 4. Return consolidated results
        return PipelineResult()
```

### Pipeline Modes

1. **Direct Mode** (`use_saved_data=False`):
   - Input → Processing → Reasoning (from DataFrame)
   - Faster for single-use scenarios
   - Data flows directly through memory

2. **Persistent Mode** (`use_saved_data=True`):
   - Input → Processing → Save to File → Reasoning (from file)
   - Better for reusable data scenarios
   - Supports data persistence and reuse

## Configuration

The system uses hierarchical configuration:

```yaml
model:
  name: "qwen2.5-7b"
  temperature: 0.2
  max_tokens: 2048

data:
  input_file: "data/sample/sales_data.csv"
  output_file: "data/output/sales_data.jsonl"
  
pipeline:
  input:
    file_uploader:
      max_file_size_mb: 100
      supported_formats: ["csv", "xlsx", "parquet", "jsonl"]
  processing:
    normalizer:
      normalize_column_names: true
      normalize_data_types: true
      imputation_strategy: "none"
  reasoning:
    reasoner:
      model_name: "qwen2.5-7b"
```

## Error Handling

The system implements comprehensive error handling:

- **Custom Exceptions**: `InputException`, `ProcessingException`, `ReasoningException`
- **Graceful Degradation**: Each module returns success/failure status
- **Error Propagation**: Errors are captured and propagated through the pipeline
- **Logging**: Comprehensive logging at each stage

## Extension Points

The modular architecture supports easy extension:

1. **New Input Formats**: Extend `BaseFileOperator`
2. **Custom Processing**: Extend `BaseProcessingOperator`
3. **Advanced Reasoning**: Extend `BaseReasoner`
4. **New Models**: Extend `BaseModel`

## Usage Example

```python
from tlp.pipeline import Pipeline

# Initialize pipeline
pipeline = Pipeline(config)

# Process data with direct reasoning
result = pipeline.process(
    input_file_path="data/sales.csv",
    output_file_path="output/processed.jsonl",
    query="What are the top 5 products by sales?",
    use_saved_data=False
)

print(f"Success: {result.success}")
print(f"Answer: {result.answer}")
```

## Development Status

**Current Version: 0.1.0**

**Implemented Features:**
- ✅ Basic file input support (CSV, Excel, Parquet, JSONL)
- ✅ Data normalization and type detection
- ✅ LLM-based reasoning with local models
- ✅ Pipeline orchestration with dual modes
- ✅ Comprehensive error handling and logging

**Planned Features:**
- 🔄 SQL tool integration
- 🔄 Python code execution tool
- 🔄 Advanced query decomposition
- 🔄 Result explanation module
- 🔄 Performance optimization

## Dependencies

- **Core**: `pandas`, `pydantic`, `pathlib`
- **File Processing**: `chardet`, `openpyxl`, `pyarrow`
- **ML/AI**: Local LLM integration
- **Utilities**: Custom logging and utility modules

This documentation reflects the current state of the TLP framework and will be updated as new features are implemented.