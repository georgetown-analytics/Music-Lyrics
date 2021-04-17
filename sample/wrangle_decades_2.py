""" Converting decades_tcc to mainframe standard """

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

def download_file_from_bucket(bucket_name, s3_key, dst_path):
    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.download_file(Key=s3_key, Filename=dst_path)

## download_file_from_bucket('music-demo-lyrics', 'lyrics_25k.csv', 'short_name.csv')
## with open('short_name.csv') as fo:
    ## print(fo.read())

decades_df = pd.read_csv('s3://worm-begin/decades_tcc_ceds_music.csv')

decades2_df = pd.DataFrame((decades_df), columns=['original_csv','artist_name','track_name','link','lyrics','language','genre','release_date'])
decades2_df['original_csv'] = 'decades_tcc'
decades3_df = decades2_df.rename(columns={'track_name':'song_name','release_date':'date'})

""" Checked for duplicates multiple ways, none overall, none by artist/song, none by lyrics.  4600 by song_name."""

"""Convert DF to csv and upload to AWS S3."""
decades3_df.to_csv('decades_df.csv', index= False)
upload_file_to_bucket('[TBD]','decades_df.csv')
