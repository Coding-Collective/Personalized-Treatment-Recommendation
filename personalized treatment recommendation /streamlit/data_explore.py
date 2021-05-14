# Importing packages
import streamlit as st
from IPython import get_ipython
from header import header
from PIL import Image
import pandas as pd  # Analysis
import warnings
import plotly.offline as py
import streamlit as st

warnings.filterwarnings('ignore')
py.init_notebook_mode(connected=True)
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


def show_image(file_name, width=1000):
    plot = Image.open(file_name)
    st.image(plot, width=width)


def data_explore():
    """
        The first page in this app made with Streamlit is for an interactive
        exploration of the continuous distributions that are available in SciPy.
        """
    header()

    @st.cache
    def load_df_train():
        df_train = pd.read_csv("../input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv", parse_dates=["date"])
        return df_train

    @st.cache
    def load_df_test():
        df_test = pd.read_csv("../input/kuc-hackathon-winter-2018/drugsComTest_raw.csv", parse_dates=["date"])

        return df_test

    df_train = load_df_train()
    df_test = load_df_test()

    col1, col2, col3 = st.beta_columns([2, 6, 1])
    with col1:
        st.write("")
    with col2:
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')
        st.text(' ')

    with col3:
        st.write("")

    st.markdown("# Let's dive in Model Exploration 📊")

    # ADD PEAK CODE
    st.markdown('<span style="color:#949494">Click to take a peak at our drug dataset 🤭</span>',
                unsafe_allow_html=True)
    if st.checkbox('', key='1'):
        st.subheader("Drug Review Dataset")
        st.write(df_test)

    st.markdown('<span style="color:#949494">Click to view all our plots 📈</span>', unsafe_allow_html=True)
    if st.checkbox('', key='2'):
        st.subheader("Plots")
        show_image(file_name='../plots/top20.png')
        show_image(file_name='../plots/bottom20.png')
        show_image(file_name='../plots/count-rating.png')
        show_image(file_name='../plots/mean-rating-day.png')
        show_image(file_name='../plots/mean-rating-month.png')
    #     rest are corrupted *fml

    st.markdown('<span style="color:#949494">Click to view all our wordclouds 🌩</span>', unsafe_allow_html=True)
    if st.checkbox('', key='3'):
        st.subheader("Plots")
        show_image(file_name='../wordcloud/review.png')
        # show_image(file_name='../wordcloud/word-count.png')
        # show_image(file_name='../wordcloud/bigram-count-plots.png')
        # show_image(file_name='../wordcloud/trigram-count-plots.png')
        # show_image(file_name='../wordcloud/4-grams-count-plots.png')
    #     rest are corrupted *fml

    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
    st.text(' ')
