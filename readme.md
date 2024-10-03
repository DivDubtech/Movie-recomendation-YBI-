# Movie Recommendation System

This project is a simple movie recommendation system that suggests similar movies based on a user's input. The system leverages TF-IDF vectorization to analyze and compare movie metadata such as genres, keywords, taglines, cast, and directors. The recommendations are provided using cosine similarity scores between movies.

## Features
- Suggest movies similar to a user’s favorite movie.
- Choose between **Top 10** or **Top 30** movie recommendations.
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into feature vectors.
- Calculates similarity between movies using cosine similarity.

## Project Structure
- `movie_recommendation.py`: The main Python script that runs the recommendation system.
- `README.md`: Documentation of the project.

## Dataset
The dataset used for this project can be found at the following link:
[Movies Recommendation Dataset](https://github.com/YBI-Foundation/Dataset/raw/a6bcba4b6f9b87d8f924df1dacad300785571cfe/Movies%20Recommendation.csv)

The dataset contains the following columns:
- `Movie_Title`: Title of the movie.
- `Movie_Genre`: Genre of the movie.
- `Movie_Keywords`: Keywords related to the movie.
- `Movie_Tagline`: Tagline of the movie.
- `Movie_Cast`: Cast of the movie.
- `Movie_Director`: Director of the movie.
- `Movie_ID`: Unique ID for each movie.

## Requirements
- Python 3.x
- pandas
- scikit-learn
- difflib (comes with Python's standard library)

### Install Dependencies

To install the required libraries, run the following command:

```bash
pip install pandas scikit-learn

