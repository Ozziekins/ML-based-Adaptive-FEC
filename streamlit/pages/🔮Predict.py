import json
import streamlit as st
import pandas as pd
import numpy as np
import time
import requests


st.title(
    """Making Prediction"""
)

df = pd.DataFrame(
   np.random.randn(24, 5),
   columns=['dropped_frames','FPS','bitrate','RTT', 'loss_rate'])

st.experimental_data_editor(df, use_container_width=True)

def on_click():
    with st.spinner('Getting Predictions'):
        url = 'http://127.0.0.1:8000/predict/json'
        resp = requests.post(url=url, json=df.to_json(orient='records'))
        if(resp.ok):
            st.header("Predictions")
            st.dataframe(get_prediction(np.asarray(json.loads(resp.json()))), use_container_width=True)
            st.success("Succesfully fetched predictions")
        else:
            st.error(resp.content)

st.button("Get Prediction", on_click=on_click)


def get_prediction(array):
    return pd.DataFrame(
        data={
            'Predicted loss rate': array,
        },
    )


