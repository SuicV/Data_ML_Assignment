import streamlit as st
import pathlib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from src.constants import RAW_DATASET_PATH, LABELS_MAP, PROCESSED_DATASET_PATH
from .page import Page

class EdaPage(Page):
    def __init__(self, name, title):
        super().__init__(name, title)
        self.name = name
        self.title = title
        self.countvectorizer = None
        if pathlib.Path(PROCESSED_DATASET_PATH).exists():
            self.df = pd.read_csv(RAW_DATASET_PATH)
            self.df['label'] = self.df['label'].map(LABELS_MAP)
            self.countvectorizer = pd.read_csv(PROCESSED_DATASET_PATH)
        else:
            self._process_resumes()

    def _process_resumes(self):
        self.df = pd.read_csv(RAW_DATASET_PATH)
        self.df['label'] = self.df['label'].map(LABELS_MAP)
        self.countvectorizer = CountVectorizer(stop_words='english')
        features = self.countvectorizer.fit_transform(self.df['resume']).toarray()
        self.countvectorizer = pd.DataFrame(features, columns=self.countvectorizer.get_feature_names_out())
        self.countvectorizer['*label*'] = self.df['label']
        self.countvectorizer.to_csv(PROCESSED_DATASET_PATH, index=False)
    
    def get_most_used_words_by_label(self, label: str) -> pd.DataFrame:
        word_means_by_category = self.countvectorizer[self.countvectorizer["*label*"] == label].\
                                    groupby("*label*", as_index=False).mean()
        word_means_by_category.drop("*label*", axis=1, inplace=True)
        word_means_by_category = word_means_by_category.T
        word_means_by_category.sort_values(by=0, ascending=False, inplace=True)
        word_means_by_category.columns = ["mean"]
        return word_means_by_category.head(25)
    
    def get_resumes_per_label(self) -> pd.DataFrame:
        return self.df.groupby("label", as_index=False).count()
    
    def display(self) -> None:
        st.header(self.title)
        st.info("In this section, we will present a basic Exploratory Data Analysis (EDA) "
                "of the resumes. Firstly, we will display the number of resumes in each role. "
                "Then, we will delve deeper into the resumes to identify the most frequently "
                "used words within each role.")
        st.subheader("Number of resumes per label")
        st.bar_chart(self.get_resumes_per_label(),
                     x="label", y="resume", color="label", height=500)
        st.warning("This graph shows that the dataset is not balanced. "
                   "Hence, the model can be biased towards the majority classes. "
                   "To overcome this issue, we can use Text Augmentation techniques "
                   "to generate new resumes in order to balance the dataset")
        role = st.selectbox(
            'Select a role',
            LABELS_MAP.values())
        if role:
            st.subheader(f"Resumes with {role} label")
            st.dataframe(self.df[self.df["label"] == role], width=780, hide_index=True)
            st.subheader(f"Most 25 used words in {role} resumes")
            st.bar_chart(self.get_most_used_words_by_label(role))
            st.success("After Analyzing the used patterns in each role. "
                       "We noticed that the most used words varies by resume label. "
                       "For instance, Web Developper role contains words related to web-dev "
                       "technologies (like: html, css, js jquery ...). On the other hand the "
                       "Oracle DBA role contains words regarding Oracle and database "
                       "(such as: 10g, oracle, database, backup). These deferences in the dataset "
                       "can be used by the ML mdoels to classify resumes")