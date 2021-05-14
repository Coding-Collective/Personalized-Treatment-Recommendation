import streamlit as st


def header():
    # Space so that 'About' box-text is lower
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")

    st.markdown("<h1 style='text-align: center;'> Welcome To </h1>",
                unsafe_allow_html=True)

    image = '../images/medicate-nobg.png'
    col1, col2, col3 = st.beta_columns([2, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.image(image, width=400)
        st.subheader('Your Personalised Treatment Recommendation ⛑')
    with col3:
        st.write("")
