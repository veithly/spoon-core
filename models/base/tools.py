from sqlalchemy import JSON, Column, String

from .base import Base


class ToolTable(Base):
    
    __tablename__ = "tools"
    
    tool_id = Column(String, primary_key=True)
    tool_name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False)
    