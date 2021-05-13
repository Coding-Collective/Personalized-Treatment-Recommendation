import streamlit as st
from PIL import Image

######################
# Header
######################

image = Image.open('images/medicate-nobg.png')

col1, mid, col2 = st.beta_columns([8,30,100])
with col1:
    st.image(image, width=350)
with col2:
    st.text(' ')
    st.text(' ')
    st.title('Personalised Treatment Recommendation')

######################
# Data exploration part - TODO: Show data related visualisations
######################
#TODO: toggle to input your own dataset
#TODO: toggle to take a peak at dataset



######################
# Model exploration - TODO: Show model related visualisations
######################

######################
# Model working - TODO: Take condition from user and display drugs
######################

######################
# End credits
######################

# st.write('')
# st.write('')
# st.write('')

st.markdown(
    '<h3 style="text-align:center;">Made with ♡ by Team <span style="font-size:35px;;font-weight:bolder"><span style="color:#4f9bce"> Rebs </span><span style="color:#FF5757"> Dels </span><span style="color:#5EE1E6"> Chels </span></span></span> 🧘🏻🏄🏻🤸‍‍</h3>',
    unsafe_allow_html=True)
