import streamlit as st
import requests
from src.constants import LABELS_MAP, SAMPLES_PATH

from .page import Page
class InferencePage(Page):
    def __init__(self, name, title):
        super().__init__(name, title)
        self.name = name
        self.title = title

    def display(self):
        st.header(self.title)
        st.info("This section simplifies the inference process. "
                "Choose a test resume and observe the label that your trained pipeline will predict."
        )
        
        sample = st.selectbox(
                    "Resume samples for inference",
                    tuple(LABELS_MAP.values()),
                    index=None,
                    placeholder="Select a resume sample",
                )
        infer = st.button('Run Inference')
        
        if infer:
            with st.spinner('Running inference...'):
                try:
                    sample_file = "_".join(sample.upper().split()) + ".txt"
                    with open(SAMPLES_PATH / sample_file, encoding="utf-8") as file:
                        sample_text = file.read()

                    result = requests.post(
                        'http://localhost:9000/api/inference',
                        json={'text': sample_text}
                    )
                    st.success('Done!')
                    label = LABELS_MAP.get(int(float(result.text)))
                    requests.post(
                        'http://localhost:9000/api/save',
                        json={
                            'resume': sample,
                            'text': sample_text,
                            'prediction': label,
                        }
                    )
                    st.metric(label="Status", value=f"Resume label: {label}")
                    st.subheader('Inference History')
                    all_inferences = requests.get('http://localhost:9000/api/inference').json()
                    st.dataframe(all_inferences)
                except Exception as e:
                    st.error('Failed to call Inference API!')
                    st.exception(e)
        pass
