from typing import List, Optional, Any
from typing_extensions import Annotated, Optional
from pydantic import BaseModel, Field, ConfigDict
from pydantic.functional_validators import BeforeValidator
from datetime import datetime

PyObjectId = Annotated[str, BeforeValidator(str)]

class AIGeneratedContent(BaseModel):
      method_name: str
      score: float


class SimilaritySource(BaseModel):
      name: str
      url: str

    
class Similarity(BaseModel):
      source: SimilaritySource
      bert_score: float
      tfidf_score: float
      score: float
      plagiarized_content: dict[str, Any]
      
      
class Document(BaseModel):
      id: Optional[PyObjectId] = Field(alias="_id", default=None)
      user_id: Optional[PyObjectId] = Field(alias="user_id", default=None)
      name: str
      path: str
      md_path: Optional[str] = None
      ai_content_result: List[AIGeneratedContent]
      similarity_result: List[Similarity]
      upload_date: datetime

      model_config = ConfigDict()
      model_config["from_attributes"] = True


class DocumentResponse(BaseModel):
    documents: List[Document]
    
    model_config = ConfigDict()
    model_config["from_attributes"] = True