from pydantic import BaseModel

class Resume(BaseModel):
    text: str

class Inference(BaseModel):
    resume: str
    text: str
    prediction: str