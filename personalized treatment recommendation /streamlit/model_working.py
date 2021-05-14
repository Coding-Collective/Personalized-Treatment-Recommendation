# Package imports
import streamlit as st
from header import header


def model_working():
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
    with col1:
        st.write("")