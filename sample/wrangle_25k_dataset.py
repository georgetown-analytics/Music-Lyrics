""" Converting two of the 25k set into a formatted to mainframe standard """

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

download_file_from_bucket('worm-begin', 'lyrics_25k.csv', 'lyrics_25k_df.csv')
with open('lyrics_25k_df.csv') as fo:
    lyrics_25k_df = pd.read_csv(fo)

"""Order and add columns"""
lyrics_25k2_df =  pd.DataFrame((lyrics_25k_df), columns=['original_csv','artist','song_name','link','lyrics','language','genre'])
"""Clean up artist column"""
lyrics_25k2_df['artist'].replace(('Lyrics'), '', regex = True, inplace = True)
"""Populate original csv column"""
lyrics_25k2_df['original_csv'] = '25K'
"""Clean up column names"""
lyrics_25k3_df = lyrics_25k2_df.rename(columns={'artist':'artist_name'})
""" Lyrics came back...why?"""
lyrics_25k3_df['artist_name'].replace(('Lyrics'), '', regex = True, inplace = True)



"""New task: Get the year information added to the main 25K frame"""
download_file_from_bucket('worm-begin', 'album_details_25k.csv', 'year_df.csv')
year_df = pd.read_csv('year_df.csv')
"""Strip the  'Lyrics' off of every 'singer_name', reduce to just what I need
to pd-merge and the year column I want to add to the main csv"""
year_df['singer_name'].replace(('Lyrics'), '', regex = True, inplace = True)
year2_df = pd.DataFrame((year_df), columns=['singer_name','year'])
year2_df=year2_df.rename(columns={'singer_name':'artist_name'})
"""This list was by artist-albums-years, so artist - years came up with 
multiple years per artist.  WHen it merged, it merged for each year.  The DF grew
by a factor of 28."""

"""Merge the two, to add date to the main csv"""
Main_25k_df = pd.merge(lyrics_25k3_df, year2_df, on='artist_name')
"""Populate original_csv column, again? Why?"""
Main_25k_df['original_csv'] = '25K'

"""Upload.  'Touch' a spot in the current dir to use, first."""
Main_25K_df.to_csv('Main_25k.csv', index= False)
upload_file_to_bucket('music-demo-lyrics','Main_25k.csv')


