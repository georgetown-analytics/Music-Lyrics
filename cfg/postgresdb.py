import psycopg2 as ps
# define credentials 
​
credentials = {'POSTGRES_ADDRESS' : 'music-lyrics.cgsxezi8cfhr.us-east-2.rds.amazonaws.com', # change to your endpoint
               'POSTGRES_PORT' : '5432', # change to your port
               'POSTGRES_USERNAME' : 'tool', # change to your username
               'POSTGRES_PASSWORD' : 'DataScience21', # change to your password
               'POSTGRES_DBNAME' : 'postgres'} # change to your db name
# create connection and cursor    
conn = ps.connect(host=credentials['POSTGRES_ADDRESS'],
                  database=credentials['POSTGRES_DBNAME'],
                  user=credentials['POSTGRES_USERNAME'],
                  password=credentials['POSTGRES_PASSWORD'],
                  port=credentials['POSTGRES_PORT'])
​
cur = conn.cursor()
​
