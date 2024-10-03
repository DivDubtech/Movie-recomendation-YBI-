# Import Libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

# Load dataset
df = pd.read_csv('https://github.com/YBI-Foundation/Dataset/raw/a6bcba4b6f9b87d8f924df1dacad300785571cfe/Movies%20Recommendation.csv')

# View dataset info
print(df.head())
print(df.info())
print(df.describe())

# Select features for recommendation
df_features = df[['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Cast', 'Movie_Director']].fillna('')
print(df_features.shape)
print(df_features)

# Combine selected features into a single string
x = df_features['Movie_Genre'] + ' ' + df_features['Movie_Keywords'] + ' ' + df_features['Movie_Tagline'] + ' ' + df_features['Movie_Cast'] + ' ' + df_features['Movie_Director']
print(x.shape)

# Convert text to feature vectors using TF-IDF
tfidf = TfidfVectorizer()
x = tfidf.fit_transform(x)
print(x.shape)

# Calculate the cosine similarity score
Similarity_Score = cosine_similarity(x)
print(Similarity_Score.shape)

# Get movie name input from user and find the closest match
Favourite_Movie_Name = input('Enter your favourite movie name: ')
All_Movies_Title_List = df['Movie_Title'].tolist()
Movie_Recommendation = difflib.get_close_matches(Favourite_Movie_Name, All_Movies_Title_List)

if len(Movie_Recommendation) > 0:
    Close_Match = Movie_Recommendation[0]
    print('Closest match to your input:', Close_Match)
    
    # Get the index of the closest matched movie
    Index_of_Close_Match_Movie = df[df.Movie_Title == Close_Match]['Movie_ID'].values[0]
    print('Index of the closest match:', Index_of_Close_Match_Movie)
    
    # Get a list of similar movies based on the similarity score
    Recommendation_Score = list(enumerate(Similarity_Score[Index_of_Close_Match_Movie]))
    print(f'Number of movies recommended: {len(Recommendation_Score)}')

    # Sort the movies based on similarity scores
    sorted_similar_movies = sorted(Recommendation_Score, key=lambda x: x[1], reverse=True)
    
    # Display the top 30 recommended movies
    print('\nTop 30 Movies Suggested for You:\n')
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = df[df.index == index]['Movie_Title'].values[0]
        if i <= 30:
            print(i, '.', title_from_index)
            i += 1
else:
    print("Sorry, no close match found for the movie name you entered.")

# Top 10 Movie Recommendation System
Movie_name = input('\nEnter your favourite movie name for top 10 recommendations: ')
list_of_all_titles = df['Movie_Title'].tolist()
find_close_match = difflib.get_close_matches(Movie_name, list_of_all_titles)

if len(find_close_match) > 0:
    close_match = find_close_match[0]
    print('Closest match to your input:', close_match)
    
    # Get the index of the closest matched movie
    index_of_movie = df[df.Movie_Title == close_match]['Movie_ID'].values[0]
    
    # Get a list of similar movies based on the similarity score
    Recommendation_Score = list(enumerate(Similarity_Score[index_of_movie]))
    
    # Sort the movies based on similarity scores
    sorted_similar_movies = sorted(Recommendation_Score, key=lambda x: x[1], reverse=True)
    
    # Display the top 10 recommended movies
    print('\nTop 10 Movies Suggested for You:\n')
    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = df[df.Movie_ID == index]['Movie_Title'].values
        if i <= 10:
            print(i, '.', title_from_index[0])
            i += 1
else:
    print("Sorry, no close match found for the movie name you entered.")
