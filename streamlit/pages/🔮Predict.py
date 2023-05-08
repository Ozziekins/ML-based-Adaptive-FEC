import streamlit as st
import pandas as pd
import numpy as np
import time

st.title(
    """Making Prediction"""
)

df = pd.DataFrame(
   np.random.randn(30, 5),
   columns=['dropped_frames','FPS','bitrate','RTT', 'loss_rate'])

st.experimental_data_editor(df, use_container_width=True)

def on_click():
    with st.spinner('Getting Predictions'):
        time.sleep(5)     
    st.header("Predictions")
    st.dataframe(get_prediction(), use_container_width=True)
    st.success("Succesfully fetched predictions")

st.button("Get Prediction", on_click=on_click)


def get_prediction():
    return pd.DataFrame(
        data={
            'Predicted loss rate': np.random.randn(30),
        },
    )


