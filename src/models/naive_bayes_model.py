from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseModel


class NaiveBayesModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline([
                    ('countv', CountVectorizer()),
                    ('nbc', MultinomialNB(**kwargs))
                ])
        )
        self.name = "Naive Bayes"