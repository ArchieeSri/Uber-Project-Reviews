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
from sklearn.feature_extraction.text import CountVectorizer


import streamlit as st

st.markdown(f'<h1 style="color:#708090;font-size:32px;">{"More Graph and Table Analysis "}</h1>', unsafe_allow_html=True)
#@st.cache
def load_data(nrows):
    data = pd.read_csv('C:/ISB/Term-02/Text Analytics/Group Project/uber_reviews_itune.csv', nrows=nrows)
    return data
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

st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Sentiment Bar Graph "}</h1>', unsafe_allow_html=True)
bloblist_desc = list()

uber_descr_str=df_new['Review'].astype(str)
for row in uber_descr_str:
    blob = TextBlob(row)
    bloblist_desc.append((row,blob.sentiment.polarity, blob.sentiment.subjectivity))
    uber_polarity_desc = pd.DataFrame(bloblist_desc, columns = ['sentence','sentiment','polarity'])
 
def f(uber_polarity_desc):
    if uber_polarity_desc['sentiment'] > 0:
        val = "Positive"
    elif uber_polarity_desc['sentiment'] == 0:
        val = "Neutral"
    else:
        val = "Negative"
    return val

uber_polarity_desc['Sentiment_Type'] = uber_polarity_desc.apply(f, axis=1)



df_new['Rating_binary'] = np.where(df_new['Rating'] >= 3, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(df_new['Review'], df_new['Rating_binary'], random_state = 0)
vect = CountVectorizer(stop_words='english', ngram_range=(1,1)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
feature_names = np.array(vect.get_feature_names())
coef_index = model.coef_[0]
df1 = pd.DataFrame({'Word':feature_names, 'Coef': coef_index})
df1.sort_values('Coef')
neg = df1[df1.Coef < 0]

st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Word Cloud of Negative Coef Count Vectoriser Words LR Model Output"}</h1>', unsafe_allow_html=True)

wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(neg.Word))
plt.axis("off")
plt.imshow(wc)
plt.show()
st.pyplot()


st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Word Cloud of Positive Coef Count Vectoriser Words LR Model Output"}</h1>', unsafe_allow_html=True)

pos = df1[df1.Coef > 0]
#st.write(pos)

wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(pos.Word))
plt.axis("off")
plt.imshow(wc)
plt.show()
st.pyplot()

X_train, X_test, y_train, y_test = train_test_split(df_new['Review'], df_new['Rating_binary'],test_size = 0.20, random_state =1000)
vectoriser = TfidfVectorizer(ngram_range=(1,2))
vectoriser.fit(X_train)

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)


#st.write(pos)
def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    st.pyplot()
    
LRmodel = LogisticRegression(solver='liblinear',C = 2, max_iter = 100000,random_state=100000)
LRmodel.fit(X_train, y_train)

st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Confusion Matrix & AUC Curve"}</h1>', unsafe_allow_html=True)
model_Evaluate(LRmodel)




st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Confusion Matrix & AUC Curve from LR-Model"}</h1>', unsafe_allow_html=True)
ypred = LRmodel.predict(X_test)
feature_names_tf = np.array(vectoriser.get_feature_names())
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, ypred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()
st.pyplot()
feature_names_tf = np.array(vectoriser.get_feature_names())
coef_index_tf = LRmodel.coef_[0]
df2 = pd.DataFrame({'Word':feature_names_tf, 'Coef': coef_index_tf})
neg_tf = df2[df2.Coef < 0]

st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Word Cloud of Negative Coef TF-IDF Words LR Model Output"}</h1>', unsafe_allow_html=True)

wc1 = WordCloud(max_words = 1000 , width = 1600 , height = 800,collocations=False).generate(" ".join(neg_tf.Word))
plt.axis("off")
plt.imshow(wc1)
plt.show()
st.pyplot()

pos_tf = df2[df2.Coef > 0]
st.markdown(f'<h1 style="color:#D2691E;font-size:28px;">{"Word Cloud of Positive Coef TF-IDF Words LR Model Output"}</h1>', unsafe_allow_html=True)
wc2 = WordCloud(max_words = 1000 , width = 1600 , height = 800,collocations=False).generate(" ".join(pos_tf.Word))
plt.axis("off")
plt.imshow(wc2, interpolation = 'bilinear')
plt.show()
st.pyplot()           
            
#st.subheader('Rating Plot')
#st.write(weekly_data)
#Bar Chart
#df = pd.DataFrame(weekly_data[:492], columns = ['Rating'])
#df.hist()
#plt.show()
#st.pyplot()