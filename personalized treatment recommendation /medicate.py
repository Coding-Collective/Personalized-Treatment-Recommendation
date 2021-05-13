import streamlit as st
import pickle
from PIL import Image

######################
# Header
######################

image = Image.open('images/medicate-nobg.png')

col1, mid, col2 = st.beta_columns([80, 30, 150])
with col1:
    st.image(image, width=350)
with col2:
    st.text(' ')
    st.text(' ')
    st.title('Personalised Treatment Recommendation')
