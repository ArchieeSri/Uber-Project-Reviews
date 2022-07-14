import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import re
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
import string
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from textblob import TextBlob
from spellchecker import SpellChecker
from langdetect import detect
#from multiapp import Multiapp



#st.sidebar.title('Page: Information')

def main_page():
    st.markdown("# Main page 🎈")
    st.sidebar.markdown("# Main page 🎈")



#def page3():
 #   st.markdown("# Page 3 🎉")
  #  st.sidebar.markdown("# Page 3 🎉")

st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache
def load_data(nrows):
    data = pd.read_csv('C:/ISB/Term-02/Text Analytics/Group Project/uber_reviews_itune.csv', nrows=nrows)
    return data

st.markdown(f'<h1 style="color:#708090;font-size:32px;">{"Welcome to the Uber Review Analysis "}</h1>', unsafe_allow_html=True)
#st.title('Main Column: Dynamic')
#st.write('Hello, this is a test Streamlit application for Uber Review.')
#st.write('It selects Review & Rating Columns  and plots the histograms.')
#st.sidebar.title('Sidebar Title: Static')


df=load_data(1000)

ax = df.groupby('Rating').count().plot(kind='bar', title='Distribution of data',legend=True)
ax.set_xticklabels(['Worst','Worse','Neutral','Good','Best'], rotation=0)
# Storing data in lists.

text, sentiment = list(df['Review']), list(df['Rating'])

    
weekly_data = load_data(1000)
#st.subheader(f'<h1 style="color:#708090;font-size:32px;">{"Data File of Uber Review Analysis "}</h1>')
st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Data File of Uber Review Analysis "}</h1>', unsafe_allow_html=True)
st.write(weekly_data)


df=load_data(1000)
st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Barplot of Rating "}</h1>', unsafe_allow_html=True)

ax = df.groupby('Rating').count().plot(kind='bar', title='Distribution of data',legend=True)
ax.set_xticklabels(['Worst','Worse','Neutral','Good','Best'], rotation=0)
#plt.figure(figsize=(100,50)) 
#import seaborn as sns
sns.countplot(x='Rating', data=df)
plt.show()
st.pyplot()
#Bar Chart
#st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Barplot of Rating "}</h1>', unsafe_allow_html=True)
#df = pd.DataFrame(weekly_data[:492], columns = ['Rating'])
#df.hist()
#plt.show()
#st.pyplot()


