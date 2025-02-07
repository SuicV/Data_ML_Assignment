import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseModel


class XGBCModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline([
                ('countv', CountVectorizer(stop_words='english', min_df=10)),
                #('tfidfv', TfidfVectorizer(stop_words='english', min_df=30)),
                ('xgbc', xgb.XGBClassifier(**kwargs))])
        )
        self.name = "XGBoost"