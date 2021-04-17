""" Converting labled_lyrics_cleaned.csv to mainframe standard """

import pandas as pd
import numpy as np
import s3fs
import os
import io
import boto3

import s3fs
fs = s3fs.S3FileSystem(anon=False,key='####',secret='######')

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

def download_file_from_bucket(bucket_name, s3_key, dst_path):
    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.download_file(Key=s3_key, Filename=dst_path)

## download_file_from_bucket('music-demo-lyrics', 'lyrics_25k.csv', 'short_name.csv')
## with open('short_name.csv') as fo:
    ## print(fo.read())

last_df = pd.read_csv('s3://worm-begin/labeled_lyrics_cleaned.csv')
"""It's a biggun!!"""

"""41 duplicates, in 160K titles"""
"""Dropped duplicates, completly.  Carter mentioned that dropping a subset like:
['artist_name','song_name'] can be an issue for data integrity"""
last2_df = last_df.rename(columns={'artist':'artist_name','seq':'lyrics','song':'song_name'})
last3_df = pd.DataFrame(last2_df, columns=['original_csv','artist_name','song_name','link','lyrics','language','genre','date'])
last3_df.drop_duplicates(inplace = True)
last3_df['original_csv'] = 'big_no_genre'
last3_df.to_csv('big_no_genre_df.csv', index = False)
upload_file_to_bucket('music-lyrics','big_no_genre_df.csv')
