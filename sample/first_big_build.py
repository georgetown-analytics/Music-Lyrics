"""First look at all the datasets"""

import pandas as pd
import numpy as np
import s3fs
import os
import io
import boto3

import s3fs
fs = s3fs.S3FileSystem(anon=False,key='#####',secret='######')

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

genre_df = pd.read_csv('s3://music-lyrics/merged2_genre_df.csv')
decades_df = pd.read_csv('s3://music-lyrics/decades3_df.csv')

big_df = pd.concat([genre_df, decades_df]).reset_index(drop = True)

big_df['artist_name'].replace('(/)','',regex=True, inplace = True) 
 """ 37K repeats on artist / song.  61k repeats on lyrics."""

 big_df.drop_duplicates(subset=['artist_name','song_name'], inplace = True)
 big_df.drop_duplicates(subset=['song_name'], inplace = True)
  """Went from 152K to 90K"""

 big_df.to_csv('big_df.csv', index = False)
upload_file_to_bucket('music-lyrics','big_df.csv')
