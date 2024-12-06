import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_data(ratings_path: str, movies_path: str):
    """
    Load ratings and movies data from CSV files.
    
    Args:
        ratings_path (str): Path to the ratings CSV file
        movies_path (str): Path to the movies CSV file
    
    Returns:
        tuple: Loaded ratings and movies DataFrames
    """
    ratings_data = pd.read_csv(ratings_path)
    movies_data = pd.read_csv(movies_path)
    return ratings_data, movies_data

def plot_distribution(data: list, title: str, bins: int = 100):
    """
    Create a histogram to visualize the distribution of data.
    
    Args:
        data (list): Data to plot
        title (str): Title of the plot
        xlabel (str): Label for x-axis
        ylabel (str): Label for y-axis
        bins (int, optional): Number of bins for histogram. Defaults to 200.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(np.log10(data), bins=bins, edgecolor='white')
    plt.title(title)
    plt.xlabel('$\\log_{10}$(Number of Ratings)')
    plt.ylabel('Frequency')
    plt.show()

def filter_entities_by_threshold(data: pd.DataFrame, entity_id_column: str = 'userId'):
    """
    Filter entities (users or movies) based on a minimum rating count threshold.
    
    Args:
        data (pd.DataFrame): Ratings DataFrame
        threshold (int): Minimum number of ratings required
        entity_id_column (str, optional): 'userId' or 'movieId'. Defaults to 'userId'.
    
    Returns:
        tuple: Filtered ratings DataFrame and set of qualified entity IDs
    """
    # Count ratings per entity
    entity_ratings_count = data.groupby(entity_id_column).size().to_dict()
    
    # Visualize distribution
    plot_distribution(
        list(entity_ratings_count.values()), 
        f'Distribution of Ratings w.r.t. {entity_id_column}'
    )

    threshold = int(input(f"Enter the minimum number of ratings w.r.t. {entity_id_column}: "))
    
    # Get qualified entity IDs
    qualified_entity_ids = {
        entity_id for entity_id, rating_count in entity_ratings_count.items() 
        if rating_count >= threshold
    }
    
    # Filter ratings data
    filtered_data = data[data[entity_id_column].isin(qualified_entity_ids)]
    
    return filtered_data, qualified_entity_ids, entity_ratings_count

def create_id_mapping(qualified_ids: set):
    """
    Create a mapping of original IDs to new consecutive IDs.
    
    Args:
        qualified_ids (set): Set of qualified entity IDs
    
    Returns:
        dict: Mapping of old IDs to new consecutive IDs
    """
    return {old_id: new_id for new_id, old_id in enumerate(sorted(qualified_ids))}

def prepare_recommendation_data(filtered_ratings: pd.DataFrame, filtered_movies: pd.DataFrame, movie_id_mapping: dict[int, int]):
    """
    Prepare data for recommendation system by creating a user-movie matrix.
    
    Args:
        filtered_ratings (pd.DataFrame): Filtered ratings data
        filtered_movies (pd.DataFrame): Filtered movies data
        movie_id_mapping (dict): Mapping of original to new movie IDs
    
    Returns:
        tuple: Train and test ratings matrices, updated movies metadata
    """
    # Update movie IDs in ratings and movies DataFrames
    filtered_ratings = filtered_ratings.copy()
    filtered_ratings['movieId'] = filtered_ratings['movieId'].map(movie_id_mapping).astype(int)

    filtered_movies = filtered_movies.copy()
    filtered_movies = filtered_movies[filtered_movies['movieId'].isin(list(movie_id_mapping.keys()))]
    filtered_movies['movieId'] = filtered_movies['movieId'].map(movie_id_mapping).astype(int)
    
    # Create user-movie ratings matrix
    user_movie_ratings_matrix = filtered_ratings.pivot(index='userId', columns='movieId', values='rating')
    user_movie_ratings_matrix.sort_index(axis=1, key=lambda x: x.astype(int), inplace=True)
    
    # Split into train and test sets
    train_user_ids, test_user_ids = train_test_split(user_movie_ratings_matrix.index, test_size=0.2)
    train_ratings_matrix = user_movie_ratings_matrix.loc[train_user_ids]
    test_ratings_matrix = user_movie_ratings_matrix.loc[test_user_ids]
    
    return train_ratings_matrix.to_numpy(), test_ratings_matrix.to_numpy(), filtered_movies.set_index('movieId').sort_index()

if __name__ == "__main__":
    # Paths to the dataset files (must be set by the user)
    path_to_ratings_file = 'raw/ratings.csv'
    path_to_movies_file = 'raw/movies.csv'
    
    # Load raw data
    ratings_data, movies_data = load_data(path_to_ratings_file, path_to_movies_file)
    
    # Filter users
    filtered_ratings_data, qualified_user_ids, user_ratings_count = filter_entities_by_threshold(
        ratings_data, 'userId'
    )
    
    # Filter movies
    filtered_ratings_data, qualified_movie_ids, movie_ratings_count = filter_entities_by_threshold(
        filtered_ratings_data, 'movieId'
    )
    
    # Create ID mappings
    movie_id_mapping = create_id_mapping(qualified_movie_ids)
    
    # Prepare recommendation data
    train_ratings_matrix, test_ratings_matrix, filtered_movies_data = prepare_recommendation_data(
        filtered_ratings_data, movies_data, movie_id_mapping
    )
    
    # Save processed data
    np.save('data/ratings_train.npy', train_ratings_matrix)
    np.save('data/ratings_test.npy', test_ratings_matrix)
    filtered_movies_data.to_csv('data/metadata.csv')
    
    # Print summary statistics
    print("\n--- Dataset Statistics ---")
    print(f"Original number of users: {len(user_ratings_count)}")
    print(f"Number of users after filtering: {len(qualified_user_ids)}")
    print(f"Percentage of users retained: {(len(qualified_user_ids) / len(user_ratings_count)) * 100:.2f}%")
    print(f"Original number of movies: {len(movie_ratings_count)}")
    print(f"Number of movies after filtering: {len(qualified_movie_ids)}")
    print(f"Percentage of movies retained: {(len(qualified_movie_ids) / len(movie_ratings_count)) * 100:.2f}%")
    print(f"Total number of users after filtering: {train_ratings_matrix.shape[0]}")
    print(f"Total number of movies after filtering: {train_ratings_matrix.shape[1]}")
    print(f"Train matrix size: {train_ratings_matrix.shape}")
    print(f"Test matrix size: {test_ratings_matrix.shape}")