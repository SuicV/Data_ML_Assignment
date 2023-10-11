import json
import os

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


from src.api.schemas import Resume, Inference
from src.models.xgbc_model import XGBCModel
from src.api.server import server
from src.api.constants import API_PREFIX
from src.constants import SAMPLES_PATH
from src.db.sqlite_connector import Base, get_db
# SETUP DATABASE AND MOCK MODEL
SQLALCHEMY_DATABASE_URL = "sqlite://"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app = server()
app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

# GET RESUME SAMPLE
sample = "Web Developer"
sample_file = "_".join(sample.upper().split()) + ".txt"
with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
    sample_text = file.read()

def test_run_inference():
    model = XGBCModel()
    with patch('src.models.xgbc_model.XGBCModel', return_value=model):
        response = client.post(API_PREFIX + "/inference", json={"text": sample_text})
    # status should be 200
    assert response.status_code == 200
    # response should be 13 (Web Developer)
    assert response.json() == 13

def test_save_inference():
    inference = Inference(text=sample_text, resume="Web Developer", prediction="Web Developer")
    response = client.post(API_PREFIX + "/save", json=json.loads(inference.model_dump_json()))
    assert response.status_code == 200
    assert response.json().get("text") == sample_text
    assert response.json().get("resume") == "Web Developer"
    assert response.json().get("prediction") == "Web Developer"

def test_get_inferences():
    response = client.get(API_PREFIX + "/inference")
    assert response.status_code == 200
    assert len(response.json()) == 1


