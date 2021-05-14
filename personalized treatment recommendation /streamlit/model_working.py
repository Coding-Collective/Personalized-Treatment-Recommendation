# Importing packages
import streamlit as st
from IPython import get_ipython
from header import header
from PIL import Image
import pandas as pd  # Analysis
import warnings
import plotly.offline as py
import streamlit as st


def show_image(file_name, width=1000):
    plot = Image.open(file_name)
    st.image(plot, width=width)


def model_working():
    header()

    @st.cache
    def load_results():
        results = pd.read_csv("../output/csv/model_result.csv")
        return results

    results = load_results()

    col1, col2, col3 = st.beta_columns([2, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
    with col1:
        st.write("")

    col1, col2, col3 = st.beta_columns([2, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.markdown("# Let's dive in Model Exploration 📈")
        # ADD PEAK CODE
        st.markdown('<span style="color:#949494">View our final model results 🤭</span>',
                    unsafe_allow_html=True)
        if st.checkbox('', key='1'):
            st.subheader("Model Results after training....")
            st.write(results)
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.subheader("🚨 Accuracy after N-gram: 64.04%")

        st.subheader("🥳 Accuracy after Lightgbm implementation: 75%")

    with col1:
        st.write("")

    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
