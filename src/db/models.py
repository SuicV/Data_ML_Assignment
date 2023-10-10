from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func
from .sqlite_connector import Base


class Inference(Base):
    __tablename__ = 'inference'

    id = Column(String, primary_key=True)
    resume = Column(String)
    text = Column(String)
    prediction = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
