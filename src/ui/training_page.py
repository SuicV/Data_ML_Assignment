import streamlit as st
from PIL import Image
from src.models.naive_bayes_model import NaiveBayesModel
from src.models.svc_model import SVCModel
from src.models.xgbc_model import XGBCModel
from src.training.train_pipeline import TrainingPipeline
from src.constants import CM_PLOT_PATH

from .page import Page

class TrainingPage(Page):
    def __init__(self, name, title):
        super().__init__(name, title)
        self.name = name
        self.title = title

    def display(self):
        st.header(self.title)
        st.info("Before you proceed to training your pipeline. Make sure you "
                "have checked your training pipeline code and that it is set properly.")
        
        model_name = st.selectbox("Choose a model to train", 
                             ("Naive Bayes", "SVM", "XGBoost"))
        if model_name == "Naive Bayes":
            model = NaiveBayesModel()
        elif model_name == "SVM":
            model = SVCModel()
        elif model_name == "XGBoost":
            model = XGBCModel()
        name = st.text_input('Pipeline name', placeholder='Naive Bayes')
        serialize = st.checkbox('Save pipeline')
        train = st.button('Train pipeline')

        if train:
            with st.spinner('Training pipeline, please wait...'):
                try:
                    tp = TrainingPipeline(model=model)
                    tp.train(serialize=serialize, model_name=name)
                    tp.render_confusion_matrix()
                    accuracy, f1 = tp.get_model_perfomance()
                    col1, col2 = st.columns(2)

                    col1.metric(label="Accuracy score", value=str(round(accuracy, 4)))
                    col2.metric(label="F1 score", value=str(round(f1, 4)))

                    st.image(Image.open(CM_PLOT_PATH), width=850)
                except Exception as e:
                    st.error('Failed to train the pipeline!')
                    st.exception(e)
        pass
