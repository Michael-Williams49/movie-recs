import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Paths to the dataset files (must be set by the user)
path_to_ratings_file = 'raw/ratings.csv'
path_to_movies_file = 'raw/movies.csv'

# Load raw data as pandas dataframes
ratings_data = pd.read_csv(path_to_ratings_file)
movies_data = pd.read_csv(path_to_movies_file)

# Count the number of ratings per user
user_ratings_count = ratings_data.groupby('userId').size().to_dict()

# Histogram for the number of ratings per user
plt.hist(user_ratings_count.values(), bins=200, edgecolor='black')
plt.title('Distribution of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.show()

# Input threshold for sufficient number of ratings per user
user_rating_threshold = int(input("Enter the minimum number of ratings a user must have: "))

# Filter users based on the threshold
qualified_user_ids = {user_id for user_id, rating_count in user_ratings_count.items() if rating_count >= user_rating_threshold}
filtered_ratings_data = ratings_data[ratings_data['userId'].isin(qualified_user_ids)]

# Count the number of ratings per movie
movie_ratings_count = filtered_ratings_data.groupby('movieId').size().to_dict()

# Histogram for the number of ratings per movie
plt.hist(movie_ratings_count.values(), bins=200, edgecolor='black')
plt.title('Distribution of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.show()

# Input threshold for sufficient number of ratings per movie
movie_rating_threshold = int(input("Enter the minimum number of ratings a movie must have: "))

# Filter movies based on the threshold
qualified_movie_ids = {movie_id for movie_id, rating_count in movie_ratings_count.items() if rating_count >= movie_rating_threshold}
filtered_ratings_data = filtered_ratings_data[filtered_ratings_data['movieId'].isin(qualified_movie_ids)]

# Calculate statistics for users
original_user_count = len(user_ratings_count)
filtered_user_count = len(qualified_user_ids)
user_retention_percentage = (filtered_user_count / original_user_count) * 100

# Calculate statistics for movies
original_movie_count = len(movie_ratings_count)
filtered_movie_count = len(qualified_movie_ids)
movie_retention_percentage = (filtered_movie_count / original_movie_count) * 100

# Create a mapping of old movie IDs to new consecutive IDs
movie_id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(qualified_movie_ids))}
filtered_ratings_data = filtered_ratings_data.copy()
filtered_ratings_data['movieId'] = filtered_ratings_data['movieId'].map(movie_id_mapping)

filtered_movies_data = movies_data[movies_data['movieId'].isin(movie_id_mapping.keys())]
filtered_movies_data = filtered_movies_data.copy()
filtered_movies_data['movieId'] = filtered_movies_data['movieId'].map(movie_id_mapping)

# Pivot the filtered ratings table into a user-movie matrix
user_movie_ratings_matrix = filtered_ratings_data.pivot(index='userId', columns='movieId', values='rating')

# Fill missing values with "NA"
user_movie_ratings_matrix = user_movie_ratings_matrix.fillna("NA")

# Split the user-movie matrix into train and test matrices
train_user_ids, test_user_ids = train_test_split(user_movie_ratings_matrix.index, test_size=0.2, random_state=42)
train_ratings_matrix = user_movie_ratings_matrix.loc[train_user_ids]
test_ratings_matrix = user_movie_ratings_matrix.loc[test_user_ids]

# Save the train, test, and metadata to CSV files
train_ratings_matrix.to_csv('data/ratings_train.csv')
test_ratings_matrix.to_csv('data/ratings_test.csv')
filtered_movies_data.set_index('movieId').sort_index().to_csv('data/metadata.csv')

# Print summary statistics
print("\n--- Dataset Statistics ---")
print(f"Original number of users: {original_user_count}")
print(f"Number of users after filtering: {filtered_user_count}")
print(f"Percentage of users retained: {user_retention_percentage:.2f}%")
print(f"Original number of movies: {original_movie_count}")
print(f"Number of movies after filtering: {filtered_movie_count}")
print(f"Percentage of movies retained: {movie_retention_percentage:.2f}%")
print(f"Total number of users after filtering: {user_movie_ratings_matrix.shape[0]}")
print(f"Total number of movies after filtering: {user_movie_ratings_matrix.shape[1]}")
print(f"Train matrix size: {train_ratings_matrix.shape}")
print(f"Test matrix size: {test_ratings_matrix.shape}")
