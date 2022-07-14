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

import streamlit as st

st.markdown("# WordCloud ❄️")
#st.sidebar.markdown("# Page 2 ❄️")




#st.set_option('deprecation.showPyplotGlobalUse', False)
#@st.cache
def load_data(nrows):
    data = pd.read_csv('C:/ISB/Term-02/Text Analytics/Group Project/uber_reviews_itune.csv', nrows=nrows)
    return data


st.markdown(f'<h1 style="color:#708090;font-size:32px;">{"Word Cloud Negative & Positive with Initial Dataset "}</h1>', unsafe_allow_html=True)

df = load_data(1000)

def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"',"'"))
    return final
import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)
#Cleaning and removing URL’s function
def cleaning_URLs(data):
    return re.sub('((www.[^s]+)|(https?://[^s]+))',' ',data)
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)
def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,'',text)
    return text


df['Review']=df['Review'].str.lower()
df['Review'] = df['Review'].apply(lambda x: cleaning_punctuations(x))
df['Review'] = df['Review'].apply(remove_punctuation)
df['Review'] = df['Review'].apply(lambda x: cleaning_URLs(x))
df['Review'] = df['Review'].apply(lambda x: cleaning_numbers(x))
df['Review'] = df['Review'].apply(lambda x: cleaning_repeating_char(x))
df['Review'] = df['Review'].apply(remove_between_square_brackets)
df['Review'] = df['Review'].apply(remove_special_characters)
df['Language']=df['Review'].apply(detect)
df_new = df[df.Language== 'en']
from spellchecker import SpellChecker

spell  = SpellChecker()
def spell_check(x):
    correct_word = []
    mispelled_word = x.split()
    for word in mispelled_word:
        correct_word.append(spell.correction(word))
    return ' '.join(correct_word)

df_new['Review'].apply(lambda x: spell_check(x))

from nltk.corpus import stopwords
stopword_list=nltk.corpus.stopwords.words('english')
STOPWORDS = set(stopwords.words('english'))
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df_new['Review'] = df_new['Review'].apply(lambda text: cleaning_stopwords(text))

stopwordlistt = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an','app','and','any','are', 'as', 'at', 'be', 'because', 'been', 'before','being', 'below', 'back', 'between','both', 'by', 'can', 'd', 'did', 'do','does', 'doing', 'down','driver', 'during', 'each','few', 'for', 'from','further', 'get','had', 'has', 'have', 'having', 'he', 'her', 'here','hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'im','in','into','is', 'it', 'its', 'itself', 'ive','just', 'll', 'm', 'ma','me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once','only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'still','such','than', 'that', "thatll", 'the', 'their', 'theirs', 'them','themselves', 'then', 'there', 'these', 'they', 'this', 'those','through', 'to', 'too','uber','under', 'until', 'up', 've', 'very','want', 'was','we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom','why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre","youve", 'your', 'yours', 'yourself', 'yourselves','hasnt','wants']  


STOPWORDS = set(stopwordlistt)
def cleaning_stopwordss(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
df_new['Review'] = df_new['Review'].apply(lambda text: cleaning_stopwordss(text))


df_new2 = df_new[df_new.Rating != 3]
df_new2['Rating_binary'] = np.where(df_new2.Rating >= 3, 1, 0)
df_new1=df_new2[['Review','Rating_binary']]
positive = df_new1[df_new1['Rating_binary'] == 1]
negative = df_new1[df_new1['Rating_binary'] == 0]

stm = nltk.PorterStemmer()
def stemming_on_text(data):
    text = [stm.stem(word) for word in data]
    return data
df_new1['Review']= df_new1['Review'].apply(lambda x: stemming_on_text(x))


lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    text = [lm.lemmatize(word) for word in data]
    return data
df_new1['Review']= df_new1['Review'].apply(lambda x: lemmatizer_on_text(x))

positive = df_new[df_new['Rating'] >= 3]
negative = df_new[df_new['Rating'] < 3]

st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Word Cloud of NEGATIVE WORDS "}</h1>', unsafe_allow_html=True)
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(negative.Review))
plt.axis("off")
plt.imshow(wc)
plt.show()
st.pyplot()

st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Word Cloud of POSITIVE WORDS "}</h1>', unsafe_allow_html=True)
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(positive.Review))
plt.axis("off")
plt.imshow(wc)
plt.show()
st.pyplot()


#st.subheader('Rating Plot')
#st.write(weekly_data)
#Bar Chart
#df = pd.DataFrame(weekly_data[:492], columns = ['Rating'])
#df.hist()
#plt.show()
#st.pyplot()