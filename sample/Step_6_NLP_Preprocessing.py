''' This notebook will perform some amount of wrangling, repeat all previous 
Natural Language Processing (NLP) preprocessing, and conduct feature engineering.  
The feature engineering leverages work conducted in Step_5_Create_Stop_and_Unique_words 
and will be used to domain-specifc scoring (like sentiment) and expanded, 
domain-specific stopwords list.

This script will have a companion notebook in the 'notebooks' 
folder of this Git repository.'''

import s3fs

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
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
from wordcloud import WordCloud, STOPWORDS #pip install
from textblob import TextBlob
!python -m textblob.download_corpora
from afinn import Afinn

import nltk
import nltk.corpus 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import preprocess_string
from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import word2vec
import multiprocessing as mp

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')
cores = mp.cpu_count()

import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

punctuation = string.punctuation + '”' + '“' + '–' + '““' + "’’" + '”'
stopword = stopwords.words('english')
stopwords = set(STOPWORDS)
wordnet_lemmatizer = WordNetLemmatizer()

#File Admin Issues

import os
import io
import boto3

from dotenv import load_dotenv
load_dotenv(verbose=True)

def aws_session(region_name='us-east-1'):
    return boto3.session.Session(aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), #looks for any .env file
                                aws_secret_access_key=os.getenv('AWS_ACCESS_KEY_SECRET'), #Has to be in same directory
                                region_name=region_name) #from above

def make_bucket(name, acl): 
    session = aws_session()
    s3_resource = session.resource('s3')
    return s3_resource.create_bucket(Bucket=name, ACL=acl)

def upload_file_to_bucket(bucket_name, file_path):
    session = aws_session()
    s3_resource = session.resource('s3')
    file_dir, file_name = os.path.split(file_path)

    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
      Filename=file_path,
      Key=file_name,
      ExtraArgs={'ACL': 'public-read'}
    )

    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{file_name}"
    return s3_url

fs = s3fs.S3FileSystem(anon=False,key='####',secret='#####')

g_df = pd.read_csv('s3://music-lyrics-chain/g_df')#entire dataset, index, song_name, lyrics, genre
g_stop = pd.read_csv('s3://music-lyrics-chain/g_stopwords')#from 80% g_train dataset, domain specific stop words
hiphop = pd.read_csv('s3://music-lyrics-chain/uniquely_hiphop')# from 80% g_train dataset, uniquely hiphop
pop = pd.read_csv('s3://music-lyrics-chain/uniquely_pop')# from 80% g_train dataset, uniquely pop
rock = pd.read_csv('s3://music-lyrics-chain/uniquely_rock')# from 80% g_train dataset, uniquely rock

g_stop = g_stop.dropna(subset=['All'])#to fix a known issue in this df
g_df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)#drop a useless column

# With appreciation for the Fake News Way
def remove_special_characters(text): 
    """
    Removes special characters from the text document
    """
    # define the pattern to keep. You can check the regex using this url https://regexr.com/
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pat, '', text)

def remove_extra_whitespace_tabs(text): 
    """
    Removes extra whitespaces and remove_extra_whitespace_tabs
    """
    #pattern = r'^\s+$|\s+$'
    pattern = r'^\s*|\s\s*'
    return re.sub(pattern, ' ', text).strip()

def remove_digits(text): 
    """
    Remove all digits from the text document
     take string input and return a clean text without numbers.
        Use regex to discard the numbers.
    """
    result = ''.join(i for i in text if not i.isdigit()).lower()
    return ' '.join(result.split())

def remove_newlines(text): 
    """
    Remove newline characters from the text document
    """
    return text.replace('\\n', ' ').replace('\\r', ' ').replace('\n', ' ').replace('\r', ' ').replace('\\', ' ')

#normalize to the NFKD (Normalization Form Compatibility Decomposition) form
#that present in the Unicode standard to remain compatible with other encodings
def remove_accented_chars(text): 
    """
    Removes accented characters from the test
    """
    new_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return new_text

import contractions
#contractions.fix(g_df['lyrics'][10])

#expands contractions found in the text
def expand_contractions(text):
    expanded_text = contractions.fix(text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# replace punctuation characters with spaces
def replace_punctuation(text):
    filters = string.punctuation + '”' + '“' + '–' + '!' + '?' + '.' + ',' #added !, ?, . , and comma
    translate_dict = dict((c, " ") for c in filters)   
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    return text

# Remove stopwords and remove words with 2 or less characters
def stops_letters(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stopword:
            result.append(token)
            
    return " ".join(result)

#Removes any word that starts with either http or https
def remove_urls(vTEXT):
    #vTEXT = re.sub('http://\S+|https://\S+', '', vTEXT,flags=re.MULTILINE)
    vTEXT = re.sub('http[s]?://\S+', '', vTEXT,flags=re.MULTILINE)
    return(vTEXT)

#Remove words that starts with www
def remove_www(vTEXT):
    vTEXT = re.sub('www\S+', '', vTEXT,flags=re.MULTILINE)
    return(vTEXT)

#Standard NLP run through.
g_df['lyrics'] = g_df['lyrics'].apply(remove_urls)
g_df['lyrics'] = g_df['lyrics'].apply(remove_www)
g_df['lyrics'] = g_df['lyrics'].apply(remove_special_characters)
g_df['lyrics'] = g_df['lyrics'].apply(remove_extra_whitespace_tabs)
g_df['lyrics'] = g_df['lyrics'].apply(remove_digits)
g_df['lyrics'] = g_df['lyrics'].apply(remove_accented_chars)
g_df['lyrics'] = g_df['lyrics'].apply(expand_contractions)
g_df['lyrics'] = g_df['lyrics'].apply(replace_punctuation)

# word counts
g_df['full_word_count'] = g_df["lyrics"].apply(lambda x: len(str(x).split(" ")))

# Character counts
g_df['full_character_count'] = g_df["lyrics"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))

#average word length
g_df['full_avg_word_length'] = g_df['full_character_count'] / g_df['full_word_count']

#Gensim stopword removal.  Creating a medium sized lyrics set.  I'll run a couple feature engineering
#functions on it.  Then create a smaller set with the domain specific stopwords list and compare the two.
g_df['med_lyrics'] =g_df['lyrics'].apply(stops_letters)

# word counts
g_df['med_word_count'] = g_df["med_lyrics"].apply(lambda x: len(str(x).split(" ")))

# Character counts
g_df['med_character_count'] = g_df["med_lyrics"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))

#average word length
g_df['med_avg_word_length'] = g_df['med_character_count'] / g_df['med_word_count']

#Feature engineering, Affinity score.

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

new_affin = get_affinity_scores(g_df['med_lyrics'].tolist())

g_df['med_content_affin'] = new_affin

#Feature engineering, Sentiment score and label

""" Something was broken in this.  The sent_score was always the same number 
and the labels were incorrect sometimes.  I fixed it with some changes however
the med_sent_score is cast as a list, an object.  Need it as a Float.

Will fix later."""

def sentiment_check (text):
    polarity_score = TextBlob(text).sentiment.polarity
    if polarity_score < 0:
        return 'negative'
    elif polarity_score == 0:
        return 'neutral'
    else:
        return 'positive'
    
g_df['med_sent_label'] = g_df['med_lyrics'].apply(sentiment_check)

print("Label done. Current Time =", datetime.now())

def new_sent_ck (text):
    polarity_score = TextBlob(text).sentiment.polarity
    sent_score = []
    sent_score.append(polarity_score)
    return sent_score

g_df['med_sent_score'] = g_df['med_lyrics'].apply(new_sent_ck) 

print("Both med_sent tasks done. Current Time =", datetime.now())

#Feature engineering, giant string for a vectorizer, later.

import nltk
nltk.download('punkt')
nltk.download('wordnet')
  
def lemmatized_word(text):

    word_tokens = nltk.word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    return  " ".join(lemmatized_word) #combine the words into a giant string that vectorizer can accept

g_df['med_vector'] = g_df['med_lyrics'].apply(lemmatized_word)

print("Vector done. Current Time =", datetime.now())

#Clean up med_lyrics for any NaN values, which will stop the next function.
g_df.dropna(axis=0, subset=['med_lyrics'], inplace=True)

#Feature engineering, create domain specific scores based on words unique to particulary genres.
def genre_count(text):
    result = 0
    text_tokenized = word_tokenize(text)
    for i in range(0, len(text_tokenized)):
        if text_tokenized[i] in stop_words:
            result += digit
        else:
            pass
    if result != 0:
        return result
    else:
        pass

#Set Rock! words...
stop_words = nltk.corpus.stopwords.words('english')

stop_words = []

rock2 = rock['Word'].to_dict()
rock3 = list(rock2.values())
digit = .01

stop_words.extend(rock3)
print(len(stop_words), 'Rock!')
print("Current Time =", datetime.now())

#Run genre_count with Rock!
g_df['med_rock_genre_count'] =g_df['med_lyrics'].apply(genre_count)

#Reset to Hip Hop...
stop_words = nltk.corpus.stopwords.words('english')

stop_words = []

hiphop2 = hiphop['Word'].to_dict()
hiphop3 = list(hiphop2.values())
digit = 100

stop_words.extend(hiphop3)
print(len(stop_words), 'Hip Hop')
print("Current Time =", datetime.now())

#Run genre_count with Hip Hop
g_df['med_hiphop_genre_count'] =g_df['med_lyrics'].apply(genre_count)

#Reset to Pop...
stop_words = nltk.corpus.stopwords.words('english')

stop_words = []

pop2 = pop['Word'].to_dict()
pop3 = list(pop2.values())
digit = 1

stop_words.extend(pop3)
print(len(stop_words), 'Pop')
print("Current Time =", datetime.now())

#Run genre_count with Hip Hop
g_df['med_pop_genre_count'] =g_df['med_lyrics'].apply(genre_count)

print("Current Time =", datetime.now())

# New coulmn with all genre_count numbers added up.
g_df['med_genre_count'] = g_df['med_rock_genre_count']+g_df['med_hiphop_genre_count']+g_df['med_pop_genre_count']

#Create the smaller lyrics set from the domain-specific stopwords.


import nltk
import nltk.corpus 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = nltk.corpus.stopwords.words('english')
g_stop2 = g_stop['All'].to_dict()
g_stop3 = list(g_stop2.values())
stop_words.extend(g_stop3)

def stops_word(text):
    result = []
    text_tokenized = word_tokenize(text)
    for i in range(0, len(text_tokenized)):
        if text_tokenized[i] not in stop_words:
            result.append(text_tokenized[i])
        else:
            pass
            
    return str(result).replace("'","")

g_df['sml_lyrics'] =g_df['lyrics'].apply(stops_word)

print("sml_lyrics complete. Current Time =", datetime.now

g_df['sml_lyrics']=g_df['sml_lyrics'].str.replace(',' ,'')# Fixes the srings with commas issue.
g_df['sml_lyrics']=g_df['sml_lyrics'].str.replace('[' ,'')
g_df['sml_lyrics']=g_df['sml_lyrics'].str.replace(']' ,'')

#Gensim stopword removal.  Same as what was run on med_lyrics.  IOT limit differences between 
#sml_ and med_ portions of dataset to just domain-specific stopwords and scoring.

g_df['sml_lyrics'] = g_df['sml_lyrics'].apply(stops_letters)

# word counts
g_df['sml_word_count'] = g_df["sml_lyrics"].apply(lambda x: len(str(x).split(" ")))

# Character counts
g_df['sml_character_count'] = g_df["sml_lyrics"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))

#average word length
g_df['sml_avg_word_length'] = g_df['sml_character_count'] / g_df['sml_word_count']

#Feature engineering, Affinity score.

afinn = Afinn()

new_affin = get_affinity_scores(g_df['sml_lyrics'].tolist())

g_df['sml_content_affin'] = new_affin

print("affinity score. Current Time =", datetime.now())

#Feature engineering, Sentiment score and label

""" Something was broken in this.  The sent_score was always the same number 
and the labels were incorrect sometimes.  I fixed it with some changes however
the sml_sent_score is cast as a list, an object.  Need it as a Float.

Will fix later."""

def sentiment_check (text):
    polarity_score = TextBlob(text).sentiment.polarity
    if polarity_score < 0:
        return 'negative'
    elif polarity_score == 0:
        return 'neutral'
    else:
        return 'positive'
    
g_df['sml_sent_label'] = g_df['sml_lyrics'].apply(sentiment_check)

print("Label done. Current Time =", datetime.now())

def new_sent_ck (text):
    polarity_score = TextBlob(text).sentiment.polarity
    sent_score = []
    sent_score.append(polarity_score)
    return sent_score

g_df['sml_sent_score'] = g_df['sml_lyrics'].apply(new_sent_ck) 

print("Both sml_sent tasks done. Current Time =", datetime.now())

#Feature engineering, giant string for a vectorizer, later.

g_df['sml_vector'] = g_df['sml_lyrics'].apply(lemmatized_word)

print("sml_vector done. Current Time =", datetime.now())

#Clean up sml_lyrics for any NaN values, which will stop the next function.
g_df.dropna(axis=0, subset=['sml_lyrics'], inplace=True)

#Feature engineering, create domain specific scores based on words unique to particulary genres.

def genre_count(text):
    result = 0
    text_tokenized = word_tokenize(text)
    for i in range(0, len(text_tokenized)):
        if text_tokenized[i] in stop_words:
            result += digit
        else:
            pass
    if result != 0:
        return result
    else:
        pass

#Set Rock! words...
stop_words = nltk.corpus.stopwords.words('english')

stop_words = []

rock2 = rock['Word'].to_dict()
rock3 = list(rock2.values())
digit = .01

stop_words.extend(rock3)
print(len(stop_words), 'Rock!')
print("Current Time =", datetime.now())

#Run genre_count with Rock!
g_df['sml_rock_genre_count'] =g_df['sml_lyrics'].apply(genre_count)

#Reset to Hip Hop...
stop_words = nltk.corpus.stopwords.words('english')

stop_words = []

hiphop2 = hiphop['Word'].to_dict()
hiphop3 = list(hiphop2.values())
digit = 100

stop_words.extend(hiphop3)
print(len(stop_words), 'Hip Hop')
print("Current Time =", datetime.now())

#Run genre_count with Hip Hop
g_df['sml_hiphop_genre_count'] =g_df['sml_lyrics'].apply(genre_count)

#Reset to Pop...
stop_words = nltk.corpus.stopwords.words('english')

stop_words = []

pop2 = pop['Word'].to_dict()
pop3 = list(pop2.values())
digit = 1

stop_words.extend(pop3)
print(len(stop_words), 'Pop')
print("Current Time =", datetime.now())

#Run genre_count with Hip Hop
g_df['sml_pop_genre_count'] =g_df['sml_lyrics'].apply(genre_count)

print("Current Time =", datetime.now())

# New coulmn with all genre_count numbers added up.
g_df['sml_genre_count'] = g_df['sml_rock_genre_count']+g_df['sml_hiphop_genre_count']+g_df['sml_pop_genre_count']

# Some final cleanup.
g_df['med_rock_genre_count'] = g_df['med_rock_genre_count'].fillna(0)
g_df['med_hiphop_genre_count'] = g_df['med_hiphop_genre_count'].fillna(0)
g_df['med_pop_genre_count'] = g_df['med_pop_genre_count'].fillna(0)
g_df['sml_rock_genre_count'] = g_df['sml_rock_genre_count'].fillna(0)
g_df['sml_hiphop_genre_count'] = g_df['sml_hiphop_genre_count'].fillna(0)
g_df['sml_pop_genre_count'] = g_df['sml_pop_genre_count'].fillna(0)

#Fix the issue with sml/med_sent_score coming out of the function as a string.
g_df['med_sent_score'] = g_df['med_sent_score'].str.replace('[' ,'')
g_df['med_sent_score'] = g_df['med_sent_score'].str.replace(']' ,'')
g_df['med_sent_score'] = pd.to_numeric(g_df['med_sent_score'], downcast='float')

g_df['sml_sent_score'] = g_df['sml_sent_score'].str.replace('[' ,'')
g_df['sml_sent_score'] = g_df['sml_sent_score'].str.replace(']' ,'')
g_df['sml_sent_score'] = pd.to_numeric(g_df['sml_sent_score'], downcast='float')

#Final reorganization to put target at front and med_genre_count in right spot.

g2_df = pd.DataFrame((g_df), columns=['genre','song_name','lyrics','full_word_count','full_character_count','full_avg_word_length',
    'med_lyrics','med_word_count','med_character_count','med_avg_word_length','med_content_affin','med_sent_label','med_sent_score',
    'med_vector','med_rock_genre_count','med_hiphop_genre_count','med_pop_genre_count','med_genre_count','sml_lyrics',
    'sml_word_count','sml_character_count','sml_avg_word_length','sml_content_affin','sml_sent_label','sml_sent_score',
    'sml_vector','sml_rock_genre_count','sml_hiphop_genre_count','sml_pop_genre_count','sml_genre_count'])
