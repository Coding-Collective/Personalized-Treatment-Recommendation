# Awesome Streamlit
import streamlit as st

# Add pages -- see those files for details within
from page_introduction import page_introduction
from data_explore import data_explore
from model_explore import model_explore
from model_working import model_working

# Use random seed
import numpy as np

np.random.seed(1)

# Set the default elements on the sidebar
# st.set_page_config(page_title='Medicate')

logo, name, temp = st.sidebar.beta_columns(3)
with logo:
    image = '../images/medicate-nobg.png'
    st.image(image, width=300)
    # st.markdown("<h3 style='text-align: left;'> Personalised Treatment Recommendation </h3>", unsafe_allow_html=True)
with name:
    pass
with temp:
    pass

st.sidebar.write(" ")


def main():
    """
    Register pages to Explore and Fit:
        page_introduction - contains page with images and brief explanations
        page_explore - contains various functions that allows exploration of
                        continuous distribution; plotting class and export
        page_fit - contains various functions that allows user to upload
                    their data as a .csv file and fit a distribution to data.
    """

    pages = {
        "What is Medicate?": page_introduction,
        "Data Exploration": data_explore,
        "Model Exploration": model_explore,
        "Model Working": model_working,
    }

    st.sidebar.subheader("Page options")

    # Radio buttons to select desired option
    page = st.sidebar.radio("Select:", tuple(pages.keys()))

    # Display the selected page with the session state
    pages[page]()

    # Write About
    st.sidebar.header("About")
    st.sidebar.warning(
        """
        Medicate — Personalised Treatment Recommendation.
        
        There are multiple drugs available for a condition, and consumers often have difficulties choosing drugs for their conditions. This recommender system potentially can help patients to choose better drugs for their conditions, and also can provide benchmark to drug providers such as doctors and pharmaceutical companies.
        """
    )

    ######################
    # End credits
    ######################

    # st.write('')
    # st.write('')
    # st.write('')

    st.markdown(
        '<h3 style="text-align:center;">Made with ♡ by Team <span style="font-size:35px;;font-weight:bolder"><span style="color:#4f9bce"> Rebs </span><span style="color:#FF5757"> Dels </span><span style="color:#5EE1E6"> Chels </span></span></span> 🧘🏻🏄🏻🤸‍‍</h3>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
