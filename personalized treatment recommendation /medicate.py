import streamlit as st
from PIL import Image


######################
# Header
######################
def show_image(file_name, width):
    plot = Image.open(file_name)
    st.image(plot, width=width)


col1, mid, col2 = st.beta_columns([8, 30, 100])
with col1:
    show_image(file_name='images/medicate-nobg.png', width=350)
with col2:
    st.text(' ')
    st.text(' ')
    st.title('Personalised Treatment Recommendation')

######################
# Table of Contents - TODO: Use <a> in st.markdown to link
######################

######################
# What is our project and some details
######################

######################
# Data exploration part - TODO: Show data related visualisations
######################
st.markdown("# Data Exploration")

# Importing packages

import pandas as pd  # Analysis
import matplotlib.pyplot as plt  # Visulization
import seaborn as sns  # Visulization
import numpy as np  # Analysis
from scipy.stats import norm  # Analysis
from sklearn.preprocessing import StandardScaler  # Analysis
from scipy import stats  # Analysis
import warnings

warnings.filterwarnings('ignore')

import gc

import os
import string

color = sns.color_palette()

from plotly import tools
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


@st.cache
def load_df_train():
    df_train = pd.read_csv("input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv", parse_dates=["date"])
    return df_train


@st.cache
def load_df_test():
    df_test = pd.read_csv("input/kuc-hackathon-winter-2018/drugsComTest_raw.csv", parse_dates=["date"])

    return df_test


df_train = load_df_train()
df_test = load_df_test()

# ADD PEAK CODE
st.markdown('<span style="color:#949494">Click to take a peak at our drug dataset 🤭</span>', unsafe_allow_html=True)
if st.checkbox('', key='1'):
    st.subheader("Drug Review Dataset")
    st.write(df_test)

st.markdown('<span style="color:#949494">Click to view all our plots 📈</span>', unsafe_allow_html=True)
if st.checkbox('', key='2'):
    st.subheader("Plots")
    show_image(file_name='plots/top20.png', width=1000)
    show_image(file_name='plots/bottom20.png', width=1000)

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
