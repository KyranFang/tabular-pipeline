import numpy as np
import pandas as pd

from pathlib import Path

from tlp.input.file_uploader import FileUploader
from tlp.processing.basic_normalizer import DataNormalizer

csv_path = "/home/fangnianrong/desktop/tabular-pipeline/data/sample/complex_sales_data.csv"
csv_path = Path(csv_path)

fileuploader = FileUploader()
result = fileuploader.process(csv_path)
print(result.data.iloc[:2])

datanormalizer = DataNormalizer(config={'imputation_strategy': 'mode'}) # Test with 'mean' imputation
result = datanormalizer.process(result.data)

print(result.data.iloc[:2])