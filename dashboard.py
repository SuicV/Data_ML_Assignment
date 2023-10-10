import streamlit as st
from src.ui.eda_page import EdaPage
from src.ui.training_page import TrainingPage
from src.ui.inference_page import InferencePage

st.title("Resume Classification Dashboard")
st.sidebar.title("Dashboard Modes")

sidebar_options = st.sidebar.selectbox(
    "Options",
    ("EDA", "Training", "Inference")
)

if sidebar_options == "EDA":
    page = EdaPage("eda", "Exploratory Data Analysis")
    page.display()
elif sidebar_options == "Training":
    page = TrainingPage("training", "Training")
    page.display()
else:
    page = InferencePage("inference", "Resume Inference")
    page.display()
