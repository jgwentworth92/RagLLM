from typing import List

from langchain_core.pydantic_v1 import BaseModel


class Sentences(BaseModel):
    sentences: List[str]