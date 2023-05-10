import streamlit as st
import pandas as pd
import requests

st.title(
    """Training Loop"""
)

st.write("In this module, you can pass files and retrain our base model to suit your needs. Currently accepts only csv files.")
uploaded_file = st.file_uploader("Choose a file", type=['csv'])


def on_click():
    if uploaded_file is not None:
        with st.spinner('Uploading dataset to the api'):
            url = 'http://127.0.0.1:8000/train'
            resp = requests.post(url=url, files={'file': uploaded_file})
            if(resp.ok):
                st.success("Successfully trained files")
            else:
                st.error("Error training files")
            print(resp.content)
    else:
        st.error("File is empty")


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
