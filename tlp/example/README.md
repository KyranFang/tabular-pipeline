# TLP Example Usage

This directory contains example configuration and run scripts for the Tabular Pipeline (TLP).

## Files

- `config.yaml` - Configuration file with model, data, and pipeline settings
- `run.py` - Main execution script
- `README.md` - This file

## Usage

1. Install dependencies:
```bash
pip install pyyaml
```

2. Activate environment:
```bash
conda activate TLP
```

3. Run the pipeline:
```bash
python tlp/example/run.py
```

## Configuration

Edit `config.yaml` to customize:
- Model settings (name, type, parameters)
- Data paths (input/output)
- Processing options
- Logging configuration
- Pipeline components

## Example Output

The script will:
1. Load the specified data file
2. Process the query using the configured model
3. Save results to the output path (if specified)
4. Log all operations

Results are saved as JSON format for easy integration with other tools.