import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class DocumentCreated(BaseModel):
    id: uuid.UUID
    chunk_count: int


class DocumentRead(BaseModel):
    id: uuid.UUID
    title: str
    content: str
    metadata: dict
    chunk_count: int
    created_at: datetime
