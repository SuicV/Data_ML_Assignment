from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseModel


class SVCModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline([
                #('countv', CountVectorizer()),
                ('tfidfv', TfidfVectorizer()),
                ('svc', SVC(**kwargs))])
        )
        self.name = "SVM"