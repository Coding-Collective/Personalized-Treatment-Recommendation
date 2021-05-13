import streamlit as st
from IPython import get_ipython
from PIL import Image


######################
# Header
######################
def show_image(file_name, width=1000):
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
    show_image(file_name='plots/top20.png')
    show_image(file_name='plots/bottom20.png')
    show_image(file_name='plots/count-rating.png')
    show_image(file_name='plots/mean-rating-day.png')
    show_image(file_name='plots/mean-rating-month.png')
#     rest are corrupted *fml

st.markdown('<span style="color:#949494">Click to view all our wordclouds 🌩</span>', unsafe_allow_html=True)
if st.checkbox('', key='3'):
    st.subheader("Plots")
    show_image(file_name='wordcloud/review.png')
    # show_image(file_name='wordcloud/word-count.png')
    # show_image(file_name='wordcloud/bigram-count-plots.png')
    # show_image(file_name='wordcloud/trigram-count-plots.png')
    # show_image(file_name='wordcloud/4-grams-count-plots.png')
#     rest are corrupted *fml

######################
# Model exploration - TODO: Show model related visualisations
######################
df_all = pd.concat([df_train, df_test])
condition_dn = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)

df_all[df_all['condition'] == '3</span> users found this comment helpful.'].head(3)

from wordcloud import WordCloud, STOPWORDS


# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(12.0, 8.0),
                   title=None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          width=400,
                          height=400,
                          mask=mask)
    wordcloud.generate(str(text))

    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,
                                   'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black',
                                   'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()


plot_wordcloud(df_all["review"], title="Word Cloud of review")

from collections import defaultdict

df_all_6_10 = df_all[df_all["rating"] > 5]
df_all_1_5 = df_all[df_all["rating"] < 6]


## custom function for ngram generation ##
def generate_ngrams(text, n_gram=1):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]


## custom function for horizontal bar chart ##
def horizontal_bar_chart(df, color):
    trace = go.Bar(
        y=df["word"].values[::-1],
        x=df["wordcount"].values[::-1],
        showlegend=False,
        orientation='h',
        marker=dict(
            color=color,
        ),
    )
    return trace


## Get the bar chart from rating  8 to 10 review ##
freq_dict = defaultdict(int)
for sent in df_all_1_5["review"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace0 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

## Get the bar chart from rating  4 to 7 review ##
freq_dict = defaultdict(int)
for sent in df_all_6_10["review"]:
    for word in generate_ngrams(sent):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'blue')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04,
                          subplot_titles=["Frequent words of rating 1 to 5",
                                          "Frequent words of rating 6 to 10"])
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")
py.iplot(fig, filename='word-plots')

freq_dict = defaultdict(int)
for sent in df_all_1_5["review"]:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'orange')

freq_dict = defaultdict(int)
for sent in df_all_6_10["review"]:
    for word in generate_ngrams(sent, 2):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(50), 'orange')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.15,
                          subplot_titles=["Frequent biagrams of rating 1 to 5",
                                          "Frequent biagrams of rating 6 to 10"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig['layout'].update(height=1200, width=1000, paper_bgcolor='rgb(233,233,233)', title="Bigram Count Plots")
py.iplot(fig, filename='word-plots')

freq_dict = defaultdict(int)
for sent in df_all_1_5["review"]:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'green')

freq_dict = defaultdict(int)
for sent in df_all_6_10["review"]:
    for word in generate_ngrams(sent, 3):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(50), 'green')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.15,
                          subplot_titles=["Frequent trigrams of rating 1 to 5",
                                          "Frequent trigrams of rating 6 to 10"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig['layout'].update(height=1200, width=1600, paper_bgcolor='rgb(233,233,233)', title="Trigram Count Plots")
py.iplot(fig, filename='word-plots')

freq_dict = defaultdict(int)
for sent in df_all_1_5["review"]:
    for word in generate_ngrams(sent, 4):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace1 = horizontal_bar_chart(fd_sorted.head(50), 'red')

freq_dict = defaultdict(int)
for sent in df_all_6_10["review"]:
    for word in generate_ngrams(sent, 4):
        freq_dict[word] += 1
fd_sorted = pd.DataFrame(sorted(freq_dict.items(), key=lambda x: x[1])[::-1])
fd_sorted.columns = ["word", "wordcount"]
trace2 = horizontal_bar_chart(fd_sorted.head(50), 'red')

# Creating two subplots
fig = tools.make_subplots(rows=1, cols=2, vertical_spacing=0.04, horizontal_spacing=0.15,
                          subplot_titles=["Frequent 4-grams of rating 1 to 5",
                                          "Frequent 4-grams of rating 6 to 10"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 1, 2)
fig['layout'].update(height=1200, width=1600, paper_bgcolor='rgb(233,233,233)', title="4-grams Count Plots")
py.iplot(fig, filename='word-plots')

rating = df_all['rating'].value_counts().sort_values(ascending=False)
rating.plot(kind="bar", figsize=(14, 6), fontsize=10, color="green")
plt.xlabel("", fontsize=20)
plt.ylabel("", fontsize=20)
plt.title("Count of rating values", fontsize=20)

cnt_srs = df_all['date'].dt.year.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14, 6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('year', fontsize=12)
plt.ylabel('', fontsize=12)
plt.title("Number of reviews in year")
plt.show()

df_all['year'] = df_all['date'].dt.year
rating = df_all.groupby('year')['rating'].mean()
rating.plot(kind="bar", figsize=(14, 6), fontsize=10, color="green")
plt.xlabel("", fontsize=20)
plt.ylabel("", fontsize=20)
plt.title("Mean rating in year", fontsize=20)

cnt_srs = df_all['date'].dt.month.value_counts()
cnt_srs = cnt_srs.sort_index()
plt.figure(figsize=(14, 6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color='green')
plt.xticks(rotation='vertical')
plt.xlabel('month', fontsize=12)
plt.ylabel('', fontsize=12)
plt.title("Number of reviews in month")
plt.show()

df_all['month'] = df_all['date'].dt.month
rating = df_all.groupby('month')['rating'].mean()
rating.plot(kind="bar", figsize=(14, 6), fontsize=10, color="green")
plt.xlabel("", fontsize=20)
plt.ylabel("", fontsize=20)
plt.title("Mean rating in month", fontsize=20)

df_all['day'] = df_all['date'].dt.day
rating = df_all.groupby('day')['rating'].mean()
rating.plot(kind="bar", figsize=(14, 6), fontsize=10, color="green")
plt.xlabel("", fontsize=20)
plt.ylabel("", fontsize=20)
plt.title("Mean rating in day", fontsize=20)

plt.figure(figsize=(14, 6))
sns.distplot(df_all["usefulCount"].dropna(), color="green")
plt.xticks(rotation='vertical')
plt.xlabel('', fontsize=12)
plt.ylabel('', fontsize=12)
plt.title("Distribution of usefulCount")
plt.show()

df_all["usefulCount"].describe()

percent = (df_all.isnull().sum()).sort_values(ascending=False)
percent.plot(kind="bar", figsize=(14, 6), fontsize=10, color='green')
plt.xlabel("Columns", fontsize=20)
plt.ylabel("", fontsize=20)
plt.title("Total Missing Value ", fontsize=20)

df_train = df_train.dropna(axis=0)
df_test = df_test.dropna(axis=0)

df_all = pd.concat([df_train, df_test]).reset_index()
del df_all['index']
percent = (df_all.isnull().sum()).sort_values(ascending=False)
percent.plot(kind="bar", figsize=(14, 6), fontsize=10, color='green')
plt.xlabel("Columns", fontsize=20)
plt.ylabel("", fontsize=20)
plt.title("Total Missing Value ", fontsize=20)

all_list = set(df_all.index)
span_list = []
for i, j in enumerate(df_all['condition']):
    if '</span>' in j:
        span_list.append(i)

new_idx = all_list.difference(set(span_list))
df_all = df_all.iloc[list(new_idx)].reset_index()
del df_all['index']

df_condition = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)
df_condition = pd.DataFrame(df_condition).reset_index()
df_condition.tail(20)

df_condition_1 = df_condition[df_condition['drugName'] == 1].reset_index()
df_condition_1['condition'][0:10]

all_list = set(df_all.index)
condition_list = []
for i, j in enumerate(df_all['condition']):
    for c in list(df_condition_1['condition']):
        if j == c:
            condition_list.append(i)

new_idx = all_list.difference(set(condition_list))
df_all = df_all.iloc[list(new_idx)].reset_index()
del df_all['index']

from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stops = set(stopwords.words('english'))

from wordcloud import WordCloud, STOPWORDS


def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(12.0, 8.0),
                   title=None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                          stopwords=stopwords,
                          max_words=max_words,
                          max_font_size=max_font_size,
                          random_state=42,
                          width=400,
                          height=400,
                          mask=mask)
    wordcloud.generate(str(text))

    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,
                                   'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black',
                                   'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()


plot_wordcloud(stops, title="Word Cloud of stops")

not_stop = ["aren't", "couldn't", "didn't", "doesn't", "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't",
            "mustn't", "needn't", "no", "nor", "not", "shan't", "shouldn't", "wasn't", "weren't", "wouldn't"]
for i in not_stop:
    stops.remove(i)

from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

stemmer = SnowballStemmer('english')


def review_to_words(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stops]
    # 6. Stemming
    stemming_words = [stemmer.stem(w) for w in meaningful_words]
    # 7. space join words
    return ' '.join(stemming_words)


get_ipython().run_line_magic('time', "df_all['review_clean'] = df_all['review'].apply(review_to_words)")

# Make a rating
df_all['sentiment'] = df_all["rating"].apply(lambda x: 1 if x > 5 else 0)

df_train, df_test = train_test_split(df_all, test_size=0.33, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

vectorizer = CountVectorizer(analyzer='word',
                             tokenizer=None,
                             preprocessor=None,
                             stop_words=None,
                             min_df=2,  # 토큰이 나타날 최소 문서 개수
                             ngram_range=(4, 4),
                             max_features=20000
                             )
vectorizer

pipeline = Pipeline([
    ('vect', vectorizer),
])

get_ipython().run_line_magic('time', "train_data_features = pipeline.fit_transform(df_train['review_clean'])")
get_ipython().run_line_magic('time', "test_data_features = pipeline.fit_transform(df_test['review_clean'])")

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
import random

# 1. Dataset
y_train = df_train['sentiment']
y_test = df_test['sentiment']
solution = y_test.copy()

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
