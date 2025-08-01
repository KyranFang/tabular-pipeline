"""全局配置设置"""

from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class GlobalSettings(BaseSettings):
    """全局配置类"""
    
    # 项目基础配置
    project_name: str = "Tabular-LLM Pipeline"
    version: str = "0.1.0"
    debug: bool = False
    
    # 路径配置
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = project_root / "data"
    logs_dir: Path = project_root / "logs"
    
    # 数据处理配置
    max_file_size_mb: int = 500
    supported_formats: list = ["csv", "tsv", "xlsx", "xls", "parquet", "jsonl"]
    supported_compressions: list = ["zip", "gz", "bz2"]
    
    # 表格处理阈值
    small_table_threshold_rows: int = 200
    small_table_threshold_cols: int = 30
    token_threshold: int = 4000
    
    # 模型配置
    default_model_path: str = "/mnt/sda/fangnianrong/hf_cache/models--Qwen--Qwen2.5-7B-Instruct"
    max_tokens: int = 2048
    temperature: float = 0.1
    
    # 日志配置
    log_level: str = "INFO"
    log_format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600
    
    class Config:
        env_prefix = "TLP_"
        case_sensitive = False


# 全局配置实例
settings = GlobalSettings()


# 确保必要目录存在
settings.data_dir.mkdir(exist_ok=True)
settings.logs_dir.mkdir(exist_ok=True)
(settings.data_dir / "raw").mkdir(exist_ok=True)
(settings.data_dir / "processed").mkdir(exist_ok=True)
(settings.data_dir / "sample").mkdir(exist_ok=True)