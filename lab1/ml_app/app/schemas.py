from pydantic import BaseModel, Field
from typing import List

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)  

class BatchTextRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    label: str
    score: float

class ModelInfoResponse(BaseModel):
    model_name: str
    num_labels: int
