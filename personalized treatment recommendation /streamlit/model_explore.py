# Package imports
import streamlit as st
from header import header
from PIL import Image


def show_image(file_name, width=1000):
    plot = Image.open(file_name)
    st.image(plot, width=width)


def model_explore():
    header()

    col1, col2, col3 = st.beta_columns([2, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')

        st.markdown("# Let's dive in Model Exploration 📊")
        show_image(file_name='../output/visualization/history.png', width=500)
    with col1:
        st.write("")

