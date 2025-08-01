import json
import uuid
import pandas as pd

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union

class BaseMetadata(BaseModel):
    pass
    
class BaseModuleOutput(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), alias='_id')
    data: Any = Field(alias='_data')
    metadata: BaseMetadata = Field(alias='_metadata')
    success: bool = Field(default=False, alias='_success')
    error_message: Optional[str] = Field(default=None, alias='_error_message')
    
    class Config:
        populate_by_name = True
        validate_by_name = True
    
    def get_id(self) -> str:
        return self.id     
    
    def get_data(self) -> Any:
        return self.data
    
    def get_metadata(self) -> BaseMetadata:
        return self.metadata
    
    def is_success(self) -> bool:
        return self.success
    
    def get_error_message(self) -> Optional[str]:
        return self.error_message
    
    # Backward compatibility properties
    @property
    def _id(self) -> str:
        return self.id
    
    @property
    def _data(self) -> Any:
        return self.data
    
    @property
    def _metadata(self) -> BaseMetadata:
        return self.metadata
    
    @property
    def _success(self) -> bool:
        return self.success
    
    @property
    def _error_message(self) -> Optional[str]:
        return self.error_message
