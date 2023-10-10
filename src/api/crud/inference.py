from sqlalchemy.orm import Session
from uuid import uuid4
from src.api import schemas
from src.db import models

def save_inference(db: Session, inference: schemas.Inference):
    db_inference = models.Inference(
        id = str(uuid4()),
        **inference.dict()
    )
    db.add(db_inference)
    db.commit()
    db.refresh(db_inference)
    return db_inference

def get_inferences(db: Session):
    return db.query(models.Inference).all()
