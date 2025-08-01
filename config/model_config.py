"""模型配置"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ModelType(str, Enum):
    """支持的模型类型"""
    LOCAL = "local"
    OPENAI = "openai"
    CUSTOM = "custom"


class ModelConfig(BaseModel):
    """模型配置基类"""
    name: str
    model_type: ModelType
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    

class LocalModelConfig(ModelConfig):
    """本地模型配置"""
    model_type: ModelType = ModelType.LOCAL
    model_path: str
    device: str = "auto"  # auto, cpu, cuda:0, etc.
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    

class OpenAIModelConfig(ModelConfig):
    """OpenAI 模型配置"""
    model_type: ModelType = ModelType.OPENAI
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    organization: Optional[str] = None
    

class CustomModelConfig(ModelConfig):
    """自定义模型配置"""
    model_type: ModelType = ModelType.CUSTOM
    model_class: str  # 模型类的完整路径
    model_kwargs: Dict[str, Any] = {}


# 预定义模型配置
PREDEFINED_MODELS = {
    "qwen2.5-7b": LocalModelConfig(
        name="Qwen2.5-7B-Instruct",
        model_path="Qwen/Qwen2.5-7B-Instruct",
        device="auto",
        max_tokens=2048,
        temperature=0.1
    ),
    
    "qwen2-7b": LocalModelConfig(
        name="Qwen2-7B",
        model_path="Qwen/Qwen2-7B",
        device="auto",
        max_tokens=2048,
        temperature=0.1
    ),
    
    "qwen3-8b": LocalModelConfig(
        name="Qwen3-8B",
        model_path="Qwen/Qwen3-8B",
        device="auto",
        max_tokens=2048,
        temperature=0.1
    ),
    
    "gpt-4": OpenAIModelConfig(
        name="gpt-4",
        max_tokens=4096,
        temperature=0.1
    ),
    
    "gpt-3.5-turbo": OpenAIModelConfig(
        name="gpt-3.5-turbo",
        max_tokens=4096,
        temperature=0.1
    )
}


def get_model_config(model_name: str) -> ModelConfig:
    if model_name not in PREDEFINED_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(PREDEFINED_MODELS.keys())}")
    return PREDEFINED_MODELS[model_name]


def register_model_config(name: str, config: ModelConfig) -> None:
    """注册新的模型配置"""
    PREDEFINED_MODELS[name] = config