import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.8

def filter_by_threshold(df: pd.DataFrame, column_name: str):
    """
    Filters a DataFrame based on the number of occurrences of values in a column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to filter on.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    value_counts = df[column_name].value_counts().to_dict()
    plt.hist(value_counts.values(), bins=100, edgecolor="white")
    plt.xlabel(f"Number of Ratings per {column_name}")
    plt.ylabel(f"Number of {column_name}s")
    plt.title(f"Distribution of Ratings per {column_name}")
    plt.show()

    threshold = int(input(f"Enter {column_name} threshold: "))
    valid_values = [value for value, count in value_counts.items() if count >= threshold]
    df_filtered = df[df[column_name].isin(valid_values)]

    return df_filtered

# 1. Load raw data
ratings_df = pd.read_csv("raw/ratings.csv")
movies_df = pd.read_csv("raw/movies.csv")

# 2. & 5. Filter users
ratings_df = filter_by_threshold(ratings_df, "userId")

# 6. Filter movies
ratings_df = filter_by_threshold(ratings_df, "movieId")

# Create ID map
movie_id_map = {old_id: new_id for new_id, old_id in enumerate(ratings_df["movieId"].unique())}
ratings_df["movieId"] = ratings_df["movieId"].map(movie_id_map)
ratings_df["movieId"] = ratings_df["movieId"].astype(int)

# 8. Map movie IDs in metadata
movies_df = movies_df[movies_df["movieId"].isin(movie_id_map.keys())]
movies_df["movieId"] = movies_df["movieId"].map(movie_id_map)
movies_df["movieId"] = movies_df["movieId"].astype(int)
movies_df.sort_values(by="movieId", axis=0, ascending=True, inplace=True)

# 9. Create rating matrix
rating_matrix = ratings_df.pivot(index="userId", columns="movieId", values="rating").sort_index(axis=1)

# 10. Split into train and test
rating_matrix = rating_matrix.to_numpy()
train_matrix, test_matrix = train_test_split(rating_matrix, train_size=TRAIN_SIZE)

# 11. Data statistics (example)
print("Original number of users:", len(ratings_df["userId"].unique()))
print("Original number of movies:", len(movies_df["movieId"].unique()))
print("Filtered number of users:", len(ratings_df["userId"].unique()))
print("Filtered number of movies:", len(ratings_df["movieId"].unique()))
print("Sparsity of rating matrix:", np.isnan(train_matrix).mean())
print("Training matrix shape: ", train_matrix.shape)
print("Test matrix shape: ", test_matrix.shape)

# 12. Save data
np.save("data/ratings_train.npy", train_matrix)
np.save("data/ratings_test.npy", test_matrix)
movies_df.to_csv("data/metadata.csv", index=False)