# app/models.py
from sqlalchemy import Column, Integer, String, Float, JSON, Text, Date, Index
from sqlalchemy import ForeignKey
from sqlalchemy.types import ARRAY
from sqlalchemy.orm import relationship
from .database import Base

class DistrictData(Base):
    __tablename__ = "district_data"
    id = Column(Integer, primary_key=True, index=True)
    state = Column(String, index=True)
    district = Column(String, index=True)
    year = Column(Integer, index=True)
    unit = Column(String, nullable=True)  # block/mandal/taluk if present
    # store original metrics as JSON for flexible columns
    metrics = Column(JSON, nullable=False)
    doc_text = Column(Text, nullable=False)  # flattened text used for embeddings
    category = Column(String, index=True)  # Safe, Semi-Critical, etc.

class EmbeddingStore(Base):
    __tablename__ = "embeddings"
    id = Column(Integer, primary_key=True, index=True)
    data_id = Column(Integer, index=True)  # DistrictData.id
    vector = Column(ARRAY(Float), nullable=False)  # we also store vector for persistence
