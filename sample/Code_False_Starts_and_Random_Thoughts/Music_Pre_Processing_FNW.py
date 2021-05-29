"""This script takes the genres dataset through a standard NLP pre-processing series of actions, 
won't call it a pipeline. It follows the path of the Fake News Cohort found here:
 (https://github.com/georgetown-analytics/From-Russia-With-Love-fake-news-/blob/master/Notebooks/Step_1_Data_Cleaning.ipynb).

 The data was wrangled previously and stored in an S3 bucket."""

import s3fs
#Boto3 File Manager at AWS S3
import os
import boto3
import io
import pandas as pd
import numpy as np
import re
import string
from datetime import datetime
import spacy
from spacy.lang import punctuation
import unicodedata 
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

punctuation = string.punctuation + '”' + '“' + '–' + '““' + "’’" + '”'
stopword = stopwords.words('english')
stopwords = set(STOPWORDS)
wordnet_lemmatizer = WordNetLemmatizer()

print("Current Time =", datetime.now())

fs = s3fs.S3FileSystem(anon=False,key='###',secret='###')
genres_df = pd.read_csv('s3://wrangled-1/merged5_genre_df.csv')

print('Download complete.')
print("Current Time =", datetime.now())

# Portions of this are excerpts from Stack Overflow responses
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
contractions.fix(genres_df['lyrics'][10])



#Expands contractions found in the text
def expand_contractions(text):

    expanded_text = contractions.fix(text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# Replace punctuation characters with spaces
def replace_punctuation(text):
    filters = string.punctuation + '”' + '“' + '–' 
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

# Removes any word that starts with either http or https
def remove_urls(vTEXT):
    #vTEXT = re.sub('http://\S+|https://\S+', '', vTEXT,flags=re.MULTILINE)
    vTEXT = re.sub('http[s]?://\S+', '', vTEXT,flags=re.MULTILINE)
    return(vTEXT)

# Remove words that starts with www
def remove_www(vTEXT):
    vTEXT = re.sub('www\S+', '', vTEXT,flags=re.MULTILINE)
    return(vTEXT)

"""Work starts below here"""
# Convert Lyrics to lowercase
genres_df['lyrics']=genres_df['lyrics'].apply(lambda x: x.lower())

# Remove URLs, www, in case they are in the lyrics
genres_df['lyrics']=genres_df['lyrics'].apply(remove_urls)
genres_df['lyrics']=genres_df['lyrics'].apply(remove_www)

# Remove special characters and extra whitespace.
genres_df['lyrics']=genres_df['lyrics'].apply(remove_special_characters)
genres_df['lyrics'] =genres_df['lyrics'].apply(remove_extra_whitespace_tabs)

#Remove digits, accented characters, contractions
genres_df['lyrics'] =genres_df['lyrics'].apply(remove_digits)
genres_df['lyrics'] =genres_df['lyrics'].apply(remove_accented_chars)
genres_df['lyrics'] =genres_df['lyrics'].apply(expand_contractions)

print('Preliminary complete.')
print("Current Time =", datetime.now())
"""A note for the Capstone project - at this point I copied the df to a csv in S3.  
While this script will continue on with Fake News method, I will follow a pre-processing 
process using spaCy as well.  The two separate here."""

# Remove punct, replace with a space. Remove stop letters 2 characters or less. Also, gensim tokenization.
genres_df['lyrics'] =genres_df['lyrics'].apply(replace_punctuation)
genres_df['lyrics'] =genres_df['lyrics'].apply(stops_letters)

"""Affinity Score - Sentiment Analysis. This uses AFINN Lexicon, which is for microblogs (tweets). This 
may not be the best sentiment lexicon for songs. spaCy uses a different one. 
Another reason to have a branch off of this trunk."""

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

new_affin = get_affinity_scores(genres_df['lyrics'].tolist())

genres_df['content_affin'] = new_affin

"""In the notebook I save the df to csv at this point."""

print('Affinity complete.')
print("Current Time =", datetime.now())

# Normalize, Lemmatize
import nltk
nltk.download('punkt')
nltk.download('wordnet')
  

def lemmatized_word(text):

    word_tokens = nltk.word_tokenize(text)
    lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
    return  " ".join(lemmatized_word) #combine the words into a giant string that vectorizer can accept

genres_df['vector'] = genres_df['lyrics'].apply(lemmatized_word)

print('Lemmatize complete.')
print("Current Time =", datetime.now())

# Add word counts
# word counts
genres_df['word_count'] = genres_df["lyrics"].apply(lambda x: len(str(x).split(" ")))

# Character counts
genres_df['character_count'] = genres_df["lyrics"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))

#average word length
genres_df['avg_word_length'] = genres_df['character_count'] / genres_df['word_count']

# Identify sentiment
def sentiment_check (text):
    polarity_score = TextBlob(text).sentiment.polarity
    genres_df['sent_score'] = polarity_score
    if polarity_score < 0:
        return 'negative'
    elif polarity_score == 0:
        return 'neutral'
    else:
        return 'positive'

genres_df['sent_label'] = genres_df['lyrics'].apply(sentiment_check)

# Two additional cleanups.  Found during further EDA.  Remove tiny genres, remove three
#songs that have NaN lyrics.

genres_df.drop(genres_df[genres_df['genre']=='Samba'].index, inplace = True)
genres_df.drop(genres_df[genres_df['genre']=='Sertanejo'].index, inplace = True)
genres_df.drop(genres_df[genres_df['genre']=='Funk Carioca'].index, inplace = True)

genres_df = genres_df.dropna(axis=0, subset=['lyrics'])

print('Saving to S3.')
print("Current Time =", datetime.now())

# Save to S3
genres_df.to_csv('genres_step_3b_df.csv', index= False)

from dotenv import load_dotenv
load_dotenv(verbose=True)

def aws_session(region_name='us-east-1'):
    return boto3.session.Session(aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                                aws_secret_access_key=os.getenv('AWS_ACCESS_KEY_SECRET'),
                                region_name=region_name)

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
    print(s3_url)

upload_file_to_bucket('music-lyrics', 'genres_step_3b_df.csv')

print("Current Time =", datetime.now())
