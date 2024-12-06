import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import infer

class Tester:
    def __init__(self, ratings_test_path: str):
        """Initialize the Tester with a model and the path to the test ratings NPY.
        
        Args:
            ratings_test_path: Path to the 'ratings_test.npy' file.
        """
        # Load the test ratings
        self.R_test: np.ndarray = np.load(ratings_test_path)
        
        self.indicators = ~np.isnan(self.R_test)
        self.R_test[~self.indicators] = 0

        self.num_users, self.num_movies = self.R_test.shape
    
    def test_top(self, model: infer.Predictor, input_size: float = 0.5, top_n: int = 30, rating_range: tuple[float, float] = (3.5, 5), verbose: bool = True, verbose_step: int = 10) -> tuple[int, float, float, float]:
        """Test the model's accuracy on the test dataset.
        
        Args:
            input_size: Proportion of observed ratings to use as input (between 0 and 1).
            top_n: Number of top recommendations to consider.
            rating_range: Tuple representing the desired rating range.
            
        Returns:
            accuracy (float between 0 and 1).
        """
        correct = 0
        total = 0
        SAE = 0
        for user_index in range(self.num_users):
            rated_movie_ids = np.argwhere(self.indicators[user_index])
            try:
                train_movie_ids, test_movie_ids = train_test_split(rated_movie_ids, train_size=input_size)
            except ValueError:
                continue
            train_movie_ids = list(train_movie_ids.flatten())
            test_movie_ids = list(test_movie_ids.flatten())

            train_ratings = {movie_id: float(self.R_test[user_index, movie_id]) for movie_id in train_movie_ids}
            recs = model.predict(train_ratings, rating_range)

            # Filter top n recs
            rec_movie_ids = list(recs.keys())

            rec_scores = list()
            for rec_id in rec_movie_ids:
                rec_scores.append(recs[rec_id][0])
            
            rec_indices = np.argsort(rec_scores)[::-1]
            rec_indices = rec_indices[:top_n]
            rec_movie_ids = [rec_movie_ids[index] for index in rec_indices]

            rec_movie_ids = np.intersect1d(rec_movie_ids, test_movie_ids)
            for movie_id in rec_movie_ids:
                actual_rating = self.R_test[user_index, movie_id]
                total += 1
                SAE += abs(actual_rating - recs[movie_id][0])
                if rating_range[0] <= actual_rating <= rating_range[1]:
                    correct += 1
            
            if verbose and (user_index + 1) % verbose_step == 0:
                print(f"User: {user_index + 1}/{self.num_users}, Total Ratings: {total}")
                print(f"  Accuracy: {(correct / total * 100):.2f}%, MAE: {SAE / total:.4g}, Average Hit: {total / (user_index + 1):.2f}")
        
        accuracy = correct / total
        MAE = SAE / total
        average_hit = total / self.num_users

        return total, accuracy, MAE, average_hit
    
    def test_all(self, model: infer.Predictor, input_size: float = 0.5, rating_range: tuple[float, float] = (3.5, 5), verbose: bool = True, verbose_step: int = 10) -> tuple[int, float, float]:
        correct = 0
        total = 0
        SAE = 0
        for user_index in range(self.num_users):
            rated_movie_ids = np.argwhere(self.indicators[user_index])
            try:
                train_movie_ids, test_movie_ids = train_test_split(rated_movie_ids, train_size=input_size)
            except ValueError:
                continue
            train_movie_ids = list(train_movie_ids.flatten())
            test_movie_ids = list(test_movie_ids.flatten())

            train_ratings = {movie_id: float(self.R_test[user_index, movie_id]) for movie_id in train_movie_ids}
            recs = model.predict(train_ratings, rating_range)

            rec_movie_ids = list(recs.keys())

            rec_movie_ids = np.intersect1d(rec_movie_ids, test_movie_ids)
            for movie_id in rec_movie_ids:
                actual_rating = self.R_test[user_index, movie_id]
                total += 1
                SAE += abs(actual_rating - recs[movie_id][0])
                if rating_range[0] <= actual_rating <= rating_range[1]:
                    correct += 1
            
            if verbose and (user_index + 1) % verbose_step == 0:
                print(f"User: {user_index + 1}/{self.num_users}, Total Ratings: {total}")
                print(f"  Accuracy: {(correct / total * 100):.2f}%, MAE: {SAE / total:.4g}")
        
        accuracy = correct / total
        MAE = SAE / total
        return total, accuracy, MAE

if __name__ == "__main__":
    # Load the trained model
    U = np.load("data/U.npy")
    V = np.load("data/V.npy")
    cov_U = np.load("data/cov_U.npy")
    cov_V = np.load("data/cov_V.npy")

    predictor = infer.Predictor(U, V, cov_U, cov_V)

    # Initialize the Tester
    tester = Tester("data/ratings_test.npy")

    total, accuracy, MAE, average_hit = tester.test_top(predictor, verbose_step=1)
    print(f"Total Ratings: {total}, Accuracy: {(accuracy * 100):.2f}%, MAE: {MAE:.4g}, Average Hit: {average_hit:.2f}")

    total, accuracy, MAE = tester.test_all(predictor, verbose_step=1)
    print(f"Total Ratings: {total}, Accuracy: {(accuracy * 100):.2f}%, MAE: {MAE:.4g}")
