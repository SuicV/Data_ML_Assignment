from http.client import HTTPException

from fastapi import APIRouter
from fastapi.params import Depends

from sqlalchemy.orm.session import Session

from src.api.schemas import Resume, Inference
from src.models.naive_bayes_model import NaiveBayesModel
from src.models.xgbc_model import XGBCModel
from src.constants import XGBOOST_PIPELINE_PATH
from src.api.crud import inference as crud
from src.db.sqlite_connector import get_db

model = XGBCModel()
model.load(XGBOOST_PIPELINE_PATH)

inference_router = APIRouter()

@inference_router.post("/inference")
def run_inference(resume: Resume):
    prediction = model.predict([resume.text])
    return prediction.tolist()[0]

@inference_router.get("/inference")
def run_inference(db: Session = Depends(get_db)):
    return crud.get_inferences(db)

@inference_router.post("/save")
def save_inference(inference: Inference, db: Session = Depends(get_db)):
    db_inference = crud.save_inference(db, inference)
    return db_inference
