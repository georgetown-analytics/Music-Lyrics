"""This script is a repository for all successful code written and tested in
the wrangling phase.  Goal is to have a single place to run script to get from 
seven unique data sets down to four with the same columns and headers.
With the .csv files pulled from the buckets.
"""

import pandas as pd
import numpy as np
import s3fs
import os
import io
import boto3

import s3fs
fs = s3fs.S3FileSystem(anon=False,key='###########',secret='##############')

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

## s3_url = upload_file_to_bucket('worm-begin','lyrics_25k.csv')
## print(s3_url) 
## s3_url = upload_file_to_bucket('worm-begin','album_details_25k.csv')
## print(s3_url)
## s3_url = upload_file_to_bucket('worm-begin','songs_details_25k.csv')
## print(s3_url)

def download_file_from_bucket(bucket_name, s3_key, dst_path):
    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.download_file(Key=s3_key, Filename=dst_path)

## download_file_from_bucket('music-demo-lyrics', 'lyrics_25k.csv', 'short_name.csv')
## with open('short_name.csv') as fo:
    ## print(fo.read())

"""I am merging the genres dataset first.  This is a large set with lyrics 
and a smaller set with the artist's genre. (Concerned that genre is 
connected to artist rather than song.)
This 'boto3' method requires a place for the csv to go in the target path. 
So I've 'touched' a few csv files in the working dir from the command line.
"""
download_file_from_bucket('worm-begin','genres_lyrics_data.csv','genres_lyrics.csv')
with open('genres_lyrics.csv') as fo:
    lyrics_df = pd.read_csv(fo)

download_file_from_bucket('worm-begin','genres_artists_data.csv','genres_genres.csv')
with open('genres_genres.csv') as fo:
    genres_df = pd.read_csv(fo)

"""Reduce genres_df to just artist-name (the key with lyrics_df) and drop dupes."""
genres1_df = pd.DataFrame(genres_df, columns=['Link','Genre'])
genres2_df = genres1_df.rename(columns={'Link':'artist_name','Genre':'genre'})
genres2_df[genres2_df.duplicated(keep = False)]

"""Reorder lyrics_df columns and rename IAW naming convention. Drop duplicates.
Drop all but the ENGLISH lyrics."""
lyrics2_df = lyrics_df.rename(columns={'ALink':'artist_name','SName':'song_name','SLink':'link','Lyric':'lyrics','Idiom':'language'})
lyrics3_df = (lyrics2_df[lyrics2_df['language']=='ENGLISH'])
lyrics3_df[lyrics3_df.duplicated(keep = False)]

"""Merge lyrics_df with genre_df to add genre to a single df with the lyrics."""

merged_genre_df = pd.merge(lyrics3_df,genres2_df,on = 'artist_name') 

"""Strip the ' / ' from 'singer_name', so we can pd.merge later (?) """
"""This is where it went wrong before!  I put in the wrong df for the replace command"""
## BAD CODING! BAD! merged2_df.replace('(/)','',regex=True, inplace = True) 
merged_genre_df['artist_name'].replace('(/)','',regex=True, inplace = True)

""" NExt need to add a column for the orginal.csv name"""
merged2_genre_df = pd.DataFrame((merged_genre_df), columns = ['original_csv','artist_name','song_name','link','lyrics','language','genre','date'])
merged2_genre_df['original_csv'] = 'genres_csv'

"""Write the dataframe to a csv.  First round I did it in a notebook, with no
path.  Locked it all up!!!!"""
"""SO.  WHen you use df.to_csv, Need to give it a path, a landing spot...
I went to command line
>>touch merged2_genre_df.csv"""
merged2_genre_df.to_csv('merged3_genre_df.csv', index=False)
""" send the merged file, cleaned up, converted to csv file to AWS S3"""
upload_file_to_bucket('[TBD]','merged3_genre_df.csv')




