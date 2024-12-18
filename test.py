import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import infer

class Tester:
    """
    A class for testing the precision and recall of a movie recommendation model.
    """
    def __init__(self, ratings_test_path: str):
        """
        Initialize the Tester with the path to the test ratings NPY.

        Args:
            ratings_test_path: Path to the 'ratings_test.npy' file.
        """
        # Load the test ratings
        self.R_test: np.ndarray = np.load(ratings_test_path)
        
        # Convert the rating matrix to binary indicators
        # self.indicators[i, j] = True if user i rated movie j, otherwise False
        self.indicators = ~np.isnan(self.R_test)
        
        # Set all NaN values to 0 for easier calculations later
        self.R_test[~self.indicators] = 0

        self.num_users, self.num_movies = self.R_test.shape
    
    def test_precision(self, model: infer.Predictor, input_size: float = 0.5, top_n: int = 30, rating_range: tuple[float, float] = (3.5, 5), verbose: bool = True, verbose_step: int = 10, baseline: bool = False) -> tuple[int, float, float]:
        """
        Test the model's precision on the test dataset.
        Precision is calculated as the proportion of correct recommendations out of the total recommendations.
        A recommendation is considered correct if the actual rating of the movie falls within the specified rating range.

        Args:
            model: The recommendation model to test.
            input_size: Proportion of observed ratings to use as input (between 0 and 1).
            top_n: Number of top recommendations to consider.
            rating_range: Tuple representing the desired rating range.
            verbose: Whether to print verbose output.
            verbose_step: The interval at which to print verbose output.

        Returns:
            A tuple containing the following:
                total: The total number of ratings considered.
                precision: The precision of the model (float between 0 and 1).
                MAE: The Mean Absolute Error of the predicted ratings.
        """
        correct = 0
        total = 0
        SAE = 0
        for user_index in range(self.num_users):
            # Get the indices of the movies rated by the current user
            rated_movie_ids = np.argwhere(self.indicators[user_index])
            
            # If the user has not rated enough movies, skip to the next user
            try:
                train_movie_ids, test_movie_ids = train_test_split(rated_movie_ids, train_size=input_size)
            except ValueError:
                continue
            
            # Flatten the train and test movie ids
            train_movie_ids = list(train_movie_ids.flatten())
            test_movie_ids = list(test_movie_ids.flatten())

            # Create a dictionary of training ratings for the user
            train_ratings = {movie_id: float(self.R_test[user_index, movie_id]) for movie_id in train_movie_ids}
            
            # Get the recommendations from the model
            if baseline:
                recs = model.random_predict(train_ratings, rating_range)
            else:
                recs = model.predict(train_ratings, rating_range)

            # Filter top n recs
            rec_movie_ids = list(recs.keys())

            # Sort the recommendations by score and select the top n movies
            rec_scores = list()
            for rec_id in rec_movie_ids:
                rec_scores.append(recs[rec_id][0])
            
            rec_indices = np.argsort(rec_scores)[::-1]
            rec_indices = rec_indices[:top_n]
            rec_movie_ids = [rec_movie_ids[index] for index in rec_indices]

            # Calculate the number of correct recommendations for the current user
            intersect_movie_ids = np.intersect1d(rec_movie_ids, test_movie_ids)
            for movie_id in intersect_movie_ids:
                actual_rating = self.R_test[user_index, movie_id]
                total += 1
                SAE += abs(actual_rating - recs[movie_id][0])
                if rating_range[0] <= actual_rating <= rating_range[1]:
                    correct += 1
            
            # Print verbose output if requested
            if verbose and ((user_index + 1) % verbose_step == 0) and (total > 0):
                print(f"User: {user_index + 1}/{self.num_users}, Total Ratings: {total}")
                print(f"  Precision: {(correct / total * 100):.2f}%, MAE: {SAE / total:.4g}")
        
        # Calculate the overall precision and MAE of the model
        precision = correct / total
        MAE = SAE / total

        return total, precision, MAE
    
    def test_recall(self, model: infer.Predictor, input_size: float = 0.5, rating_range: tuple[float, float] = (3.5, 5), verbose: bool = True, verbose_step: int = 10, baseline: bool = False) -> tuple[int, float]:
        """
        Test the model's recall on the test dataset.
        Recall is calculated as the proportion of relevant movies that are recommended,
        where a relevant movie is one that the user has rated within the specified rating range.

        Args:
            model: The recommendation model to test.
            input_size: Proportion of observed ratings to use as input (between 0 and 1).
            rating_range: Tuple representing the desired rating range.
            verbose: Whether to print verbose output.
            verbose_step: The interval at which to print verbose output.

        Returns:
            A tuple containing the following:
                total: The total number of relevant movies.
                recall: The recall of the model (float between 0 and 1).
        """
        correct = 0
        total = 0
        for user_index in range(self.num_users):
            # Get the indices of the movies rated by the current user
            rated_movie_ids = np.argwhere(self.indicators[user_index])
            
            # If the user has not rated enough movies, skip to the next user
            try:
                train_movie_ids, test_movie_ids = train_test_split(rated_movie_ids, train_size=input_size)
            except ValueError:
                continue

            train_movie_ids = list(train_movie_ids.flatten())
            test_movie_ids = list(test_movie_ids.flatten())

            # Create a dictionary of training ratings for the user
            train_ratings = {movie_id: float(self.R_test[user_index, movie_id]) for movie_id in train_movie_ids}
            
            # Get the recommendations from the model
            if baseline:
                recs = model.random_predict(train_ratings, rating_range)
            else:
                recs = model.predict(train_ratings, rating_range)

            # Calculate the number of relevant movies that are recommended
            rec_movie_ids = list(recs.keys())

            for movie_id in test_movie_ids:
                actual_rating = self.R_test[user_index, movie_id]
                if rating_range[0] <= actual_rating <= rating_range[1]:
                    total += 1
                    if movie_id in rec_movie_ids:
                        correct += 1
            
            # Print verbose output if requested
            if verbose and ((user_index + 1) % verbose_step == 0) and (total > 0):
                print(f"User: {user_index + 1}/{self.num_users}, Total Ratings: {total}, Recall: {(correct / total * 100):.2f}%")
        
        # Calculate the overall recall of the model
        recall = correct / total
        return total, recall
    
if __name__ == "__main__":
    # Load the trained model
    U = np.load("data/U.npy")
    V = np.load("data/V.npy")
    cov_U = np.load("data/cov_U.npy")
    cov_V = np.load("data/cov_V.npy")

    predictor = infer.Predictor(U, V, cov_U, cov_V)

    # Initialize the Tester
    tester = Tester("data/ratings_test.npy")

    print("== Precision Test ==")
    total, precision, MAE = tester.test_precision(predictor, verbose_step=1)
    print(f"Total Ratings: {total}, Precision: {(precision * 100):.2f}%, MAE: {MAE:.4g}")
    
    print("== Precision Baseline ==")
    total, precision, MAE = tester.test_precision(predictor, verbose=False, baseline=True)
    print(f"Total Ratings: {total}, Precision: {(precision * 100):.2f}%, MAE: {MAE:.4g}")

    print("== Recall Test ==")
    total, recall = tester.test_recall(predictor, verbose_step=1)
    print(f"Total Ratings: {total}, Recall: {(recall * 100):.2f}%")

    print("== Recall Baseline ==")
    total, recall = tester.test_recall(predictor, verbose=False, baseline=True)
    print(f"Total Ratings: {total}, Recall: {(recall * 100):.2f}%")