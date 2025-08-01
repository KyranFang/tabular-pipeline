import torch

from typing import Any, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

from tlp.reasoning.base import BaseModel
from config.model_config import LocalModelConfig
from tlp.exceptions import ModelLoadException, InferenceException

class LocalModel(BaseModel):
    def __init__(self, config: LocalModelConfig):
        super().__init__(config.dict())
        self.model_config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self.load_model()
    
    def load_model(self) -> bool:
        self.logger.info(f"Loading model: {self.model_config.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self._loaded = True

    
    def is_loaded(self) -> bool:
        return self._loaded and self.model is not None
    
    def generate(self, prompt: str) -> str:
        if not self.is_loaded():
            raise InferenceException("Model not loaded")
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate with basic configuration
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.model_config.max_tokens,
                temperature=self.model_config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return generated_text