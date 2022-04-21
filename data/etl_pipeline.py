import pandas as pd
from sqlalchemy import create_engine


# Load  datasets
messages = pd.read_csv('messages.csv')
categories = pd.read_csv('categories.csv')

# Merge dataframes
df = messages.merge(categories, left_on='id', right_on='id')

# Grab the desaster categories and separate in individual tables
categories =  df['categories'].str.split(';', expand=True)
row = categories.iloc[0]
row = [x.split('-')[0] for x in row]
category_colnames = row
categories.columns = category_colnames

# Clean the categories columns to let just the number
for column in categories:
    categories[column] = [x.split('-')[-1] for x in categories[column]]
    categories[column] = categories[column].astype(int)

# drop old categories' column from the main dataframe
df.drop(['categories'], axis=1, inplace = True)

# Add id column to the new categories dataframe to merge them
categories['id'] = df['id']
df = messages.merge(categories, left_on='id', right_on='id')

# Drop duplicates
df = df.drop_duplicates()

# Save into a table in SQLlite
engine = create_engine('sqlite:///desaster_project.db')
df.to_sql('disaster_message', engine, index=False)