import streamlit as st
import time
import pandas as pd
import uuid

st.title(
    """Training Loop"""
)

st.write("In this module, you can pass files and retrain our base model to suit your needs. Currently accepts only csv files.")
uploaded_file = st.file_uploader("Choose a file",
                                 type=['csv'],
                                 accept_multiple_files=True)


def on_click():
    with st.spinner('Uploading dataset to the api'):
        if 'intraining' not in st.session_state:
            st.session_state['intraining'] = uuid.uuid4()
            time.sleep(5)
        else:
            st.error("A Training Process is still occurring")


st.button("Start Training", on_click=on_click)

st.header("Metrics")


def get_metrics():
    return pd.DataFrame(
        data={
            'description': ['Lorem Ipsum do ', 'Lorem ipsum do', 'Lorem ipsum do', 'Lorem ipsum do', ''],
            'Values': [0, 1, 2, 3, 4],
        },
        index=['Metric 1', 'Metric 2', 'Metric 3', 'Metric 4', 'Metric 5']
    )


st.dataframe(get_metrics(), use_container_width=True)
