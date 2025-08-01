import chardet
import pandas as pd


from io import StringIO
from pathlib import Path


def dataframe_to_string(data: pd.DataFrame, max_rows: int = 100, max_cols: int = 20) -> str:
    """Convert DataFrame to string representation"""
    # Limit displayed rows and columns
    display_data = data.head(max_rows)
    if len(data.columns) > max_cols:
        display_data = display_data.iloc[:, :max_cols]
    
    # Convert to string
    buffer = StringIO()
    display_data.to_string(buf=buffer, index=False, max_rows=max_rows)
    table_str = buffer.getvalue()
    
    # Add truncation information
    if len(data) > max_rows or len(data.columns) > max_cols:
        table_str += f"\n\n[Note: Table truncated for display, original data has {len(data)} rows and {len(data.columns)} columns]"
    
    return table_str

def get_file_extension(file_path: Path) -> str:
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    
    if file_path.is_dir():
        return ""
    
    supported_compressions = ['zip', 'gz', 'bz2']
    suffixes = [s.lower() for s in file_path.suffixes]
    if len(suffixes) >= 2 and suffixes[-1][1:] in {c.lower() for c in supported_compressions}:
        return suffixes[-2][1:]
    if suffixes:
        return suffixes[-1][1:]
    return ""

def detect_encoding(file_path: Path) -> str:
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000) 
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0)
            
            # Debug: Detected encoding with confidence
            if confidence < 0.7:
                encoding = 'utf-8'
            
            return encoding
    except Exception as e:
        # Warning: Encoding detection failed, using utf-8
        return 'utf-8'
    
def decompress_file(file_path: Path) -> Path:   
    compression = file_path.suffix[1:].lower()
    supported_compressions = ['zip', 'gz', 'bz2']
    
    if compression not in supported_compressions:
        return file_path
    
    from config.settings import settings
    temp_dir = settings.data_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    
    if compression == 'gz':
        decompressed_name = file_path.stem
    else:
        decompressed_name = file_path.stem
    
    decompressed_path = temp_dir / decompressed_name
    
    try:
        import zipfile, gzip, bz2
        from tlp.exceptions import FileFormatException
        
        # TODO: Multi-file support
        if compression == 'zip':
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                names = zip_ref.namelist()
                if len(names) != 1:
                    raise FileFormatException(f"ZIP file should contain exactly one file, found {len(names)}")
                
                zip_ref.extract(names[0], temp_dir)
                decompressed_path = temp_dir / names[0]
        
        elif compression == 'gz':
            with gzip.open(file_path, 'rb') as gz_file:
                with open(decompressed_path, 'wb') as out_file:
                    out_file.write(gz_file.read())
        
        elif compression == 'bz2':
            with bz2.open(file_path, 'rb') as bz2_file:
                with open(decompressed_path, 'wb') as out_file:
                    out_file.write(bz2_file.read())
        
        return decompressed_path
        
    except Exception as e:
        raise FileFormatException(f"Failed to decompress {file_path}: {e}")
    
def load_csv(file_path: Path, encoding: str) -> pd.DataFrame:
    try:
        separators = [',', '\t', ';', '|']
        
        for sep in separators:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=sep, nrows=5)
                if len(df.columns) > 1:
                    return pd.read_csv(file_path, encoding=encoding, sep=sep)
            except:
                continue
        
        return pd.read_csv(file_path, encoding=encoding)
    except Exception as e:
        raise FileFormatException(f"Failed to load CSV file {file_path}: {e}")
    
def load_excel(file_path: Path) -> pd.DataFrame:
    try:
        excel_file = pd.ExcelFile(file_path)
        
        if len(excel_file.sheet_names) == 0:
            raise FileFormatException("Excel file contains no sheets")
        
        sheet_name = excel_file.sheet_names[0]
        # Info: Reading Excel sheet
        
        return pd.read_excel(file_path, sheet_name=sheet_name)
        
    except Exception as e:
        raise FileFormatException(f"Failed to load Excel file {file_path}: {e}")
    
def load_parquet(file_path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        raise FileFormatException(f"Failed to load Parquet file {file_path}: {e}")

def load_jsonl(file_path: Path, encoding: str) -> pd.DataFrame:
    try:
        return pd.read_json(file_path, lines=True, encoding=encoding)
    except Exception as e:
        raise FileFormatException(f"Failed to load JSONL file {file_path}: {e}")