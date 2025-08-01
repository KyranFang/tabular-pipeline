import os
import sys
import yaml
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tlp.pipeline import Pipeline
from tlp.utils.logger import get_logger

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Get config path
    config_path = Path(__file__).parent / "config.yaml"
    config = load_config(config_path)
    logger = get_logger(__name__)
    logger.info(f"Starting TLP with config: {config_path}")
    pipeline = Pipeline(config=config)
    input_data_path = project_root / config['data']['input_file_path']
    output_data_path = project_root / config['data']['output_file_path']
    query = "What is the sales amount in the first row?"
    
    logger.info(f"Processing data from: {input_data_path}")
    logger.info(f"Query: {query}")
    
    result = pipeline.process(str(input_data_path), str(output_data_path), query)
    logger.info(f"Processing completed successfully")
    logger.info(f"Answer: {result.answer}")
    
    # Save output if specified
    if config['data'].get('result_file_path'):
        output_path = project_root / config['data']['result_file_path']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()