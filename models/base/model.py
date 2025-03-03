import json
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import (JSON, Column, DateTime, ForeignKey, String,
                        func)
from sqlalchemy.orm import relationship

from .base import Base


class LLM(BaseModel):
    name: str = Field(default_factory=str)
    max_tokens: int = Field(default=500)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)


class Chain(BaseModel):
    llm: LLM
    prompt: str = Field(default_factory=str)


class Model(BaseModel):
    id: Optional[str] = Field(default="")
    chains: Optional[list[Chain]] = Field(default=[])
    owner_address: Optional[str] = Field(default="")


class ModelTable(Base):
    __tablename__ = "models"
    id = Column(String, primary_key=True)
    chains = Column(JSON, default=list, nullable=False)
    sessions = relationship("SessionStateTable", back_populates="model", cascade="all, delete-orphan")
    owner_address = Column(String, nullable=False)
    
    def __init__(self, id: str, chains: list[Chain], owner_address: str):
        self.id = id
        self.chains = [chain.model_dump() for chain in chains]
        self.owner_address = owner_address

    def model_dump_json(self) -> str:
        return json.dumps({
            "id": self.id,
            "chains": self.chains,
            "owner_address": self.owner_address
        })

    @staticmethod
    def model_validate_json(json_str: str) -> Model:
        return Model.model_validate_json(json_str)

class SessionStateTable(Base):
    __tablename__ = "sessions"
    
    session_id = Column(String, primary_key=True)
    model_id = Column(String, ForeignKey("models.id"), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    owner_address = Column(String, nullable=False)
    
    model = relationship("ModelTable", back_populates="sessions")
    chat_history = Column(JSON, default=list, nullable=False)
