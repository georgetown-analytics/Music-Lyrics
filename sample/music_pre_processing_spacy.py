"""
This 3A notebook will be perform the second half of NLP preprocessing using spaCy, mostly. 
Also, this will produce a large dataframe two lyric sets (with/with out stopwords), 
each with word counts and spaCy .doc (vectorization, POS tags, Named Entity recognition, etc.).

Will run spaCy and append the nlp.doc, on the (nearly) complete lyrics. 
Add word / letter counts to the df, run affinity and sentiment append to df.
Then will remove stopwords. And do spaCy, counts, affinity, sentiment.

At the end of this NB we will have four possible paths to run EDA or modeling 
IOT see impacts of various pre-processing choices:
1) Semi-clean, full lyrics. 
2) Fully clean, NLTK NLP pre-processing (affinity and sentiment). 
3) Semi-clean, full lyrics, spaCy vectorization and NLTK pre-processing. 
4) Fully clean, spaCy vectorization and NLTK pre-processing.
"""

import s3fs
import pandas as pd
pd.set_option('display.max_columns', 100000)
pd.set_option('display.max_row', 1000000)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
# import tldextract Accurately separate the TLD from the registered domain and subdomains of a URL
from tqdm.autonotebook import tqdm
tqdm.pandas(desc="progress-bar", leave=False)
import string

import spacy
from spacy.lang import punctuation
from spacy.lang.en import English
from spacy import displacy
nlp = spacy.load("en_core_web_lg")

import unicodedata  # might need to pip install unicodedate2 on aws sagemaker
import contractions
from contractions import contractions_dict ## pip installed this
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import STOPWORDS
import warnings
from afinn import Afinn
warnings.filterwarnings('ignore')

%matplotlib inline
sns.set(style='darkgrid',palette='Dark2',rc={'figure.figsize':(9,6),'figure.dpi':90})

punctuation = string.punctuation + '”' + '“' + '–' + '““' + "’’" + '”'
stopword = stopwords.words('english')
stopwords = set(STOPWORDS)
wordnet_lemmatizer = WordNetLemmatizer()

"""
Pulling down the dataset which completed some, but not all, of the NLP preprocessing in NB 3 
(Using the Fake News group's pre-processing notebook as a guide 
(https://github.com/georgetown-analytics/From-Russia-With-Love-fake-news-/blob/master/Notebooks/Step_1_Data_Cleaning.ipynb)).

This particular starter set has been through: lowercase, remove URLs, ww, special characters, 
extra whitespace, accented characters and expanded contractions.
"""

#fs = s3fs.S3FileSystem(anon=False,key='###',secret='###')

#g_df = pd.read_csv('s3://music-lyrics-chain/genres_midcln_df.csv')

"""
Two additional cleanups.  Found during further EDA.  Remove tiny genres, remove three
songs that have NaN lyrics.
"""

g_df.drop(g_df[g_df['genre']=='Samba'].index, inplace = True)
g_df.drop(g_df[g_df['genre']=='Sertanejo'].index, inplace = True)
g_df.drop(g_df[g_df['genre']=='Funk Carioca'].index, inplace = True)

g_df = g_df.dropna(axis=0, subset=['lyrics'])

#Run spaCy.nlp on full lyrics.  Took over an hour.  Grew df to ~250meg
lyrics = [_ for _ in g_df['lyrics']]

def set_doc(lyrics):
    scores = []
    for t in lyrics:
        doc = nlp(t)
        scores.append(doc)
    return scores

new_doc = set_doc(g_df['lyrics'].tolist())

g_df['spaCy_Doc'] = new_doc

# Affinity scoring, full lyrics.

afinn = Afinn()

def get_affinity_scores(lyrics):
    scores = []
    count = 0
    for t in lyrics:
        if len(t) > 0:
            scores.append(afinn.score(t) / len(t))
        else:
            count += 1
            scores.append(0)
    return scores

new_affin = get_affinity_scores(g_df['lyrics'].tolist())
g_df['content_affin'] = new_affin

#Word / Character counts

# word counts
g_df['word_count'] = g_df["lyrics"].apply(lambda x: len(str(x).split(" ")))

# Character counts
g_df['character_count'] = g_df["lyrics"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))

#average word length
g_df['avg_word_length'] = g_df['character_count'] / g_df['word_count']

#Add NLTK vectorization.  Keeps the spaCy data aligned with FNW.

import nltk
nltk.download('punkt')
nltk.download('wordnet')
  

def lemmatized_word(text):

    word_tokens = nltk.word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    return  " ".join(lemmatized_word) #combine the words into a giant string that vectorizer can accept

g_df['vector'] = g_df['lyrics'].apply(lemmatized_word)

#Using affinity score from before, run a sentiment check on the full dataset.

def sentiment_check (text):
    polarity_score = TextBlob(text).sentiment.polarity
    g_df['sent_score'] = polarity_score
    if polarity_score < 0:
        return 'negative'
    elif polarity_score == 0:
        return 'neutral'
    else:
        return 'positive'
    
g_df['sent_label'] = g_df['lyrics'].apply(sentiment_check)

#Creating the _sml version of the lyrics, the spaCy_Doc column, affinity, the word counts, sentiment scores.

def replace_punctuation(text):
    filters = string.punctuation + '”' + '“' + '–' 
    translate_dict = dict((c, " ") for c in filters)   
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    return text

def stops_letters(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stopword:
            result.append(token)
            
    return " ".join(result)

g_df['lyrics_sml'] =g_df['lyrics'].apply(replace_punctuation)
g_df['lyrics_sml'] =g_df['lyrics'].apply(stops_letters)

# Run spaCy on lyrics_sml.

lyrics = [_ for _ in g_df['lyrics_sml']]

def set_doc(lyrics):
    scores = []
    for t in lyrics:
        doc = nlp(t)
        scores.append(doc)
    return scores

new_doc = set_doc(g_df['lyrics_sml'].tolist())

g_df['spaCy_Doc_sml'] = new_doc

# Run affinity on lyrics_sml

new_affin = get_affinity_scores(g_df['lyrics_sml'].tolist())
g_df['content_affin_sml'] = new_affin

# word counts
g_df['word_count_sml'] = g_df['lyrics_sml'].apply(lambda x: len(str(x).split(" ")))

# Character counts
g_df['character_count_sml'] = g_df['lyrics_sml'].apply(lambda x: sum(len(word) for word in str(x).split(" ")))

#average word length
g_df['avg_word_length_sml'] = g_df['character_count_sml'] / g_df['word_count_sml']

# NLTK vector on lyrics_sml

g_df['vector_sml'] = g_df['lyrics_sml'].apply(lemmatized_word)

# Sentiment score and classification from affinity score done earlier.

g_df['sent_label_sml'] = g_df['lyrics_sml'].apply(sentiment_check)




