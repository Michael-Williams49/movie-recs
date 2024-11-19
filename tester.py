import numpy as np
import pandas as pd
from typing import Union
from infer import Normal_Joint, Feature_Joint

class Tester:
    def __init__(self, model_to_test: Union[Normal_Joint, Feature_Joint], ratings_test_path: str):
        """Initialize the Tester with a model and the path to the test ratings CSV.
        
        Args:
            model_to_test: An instance of Normal_Joint or Feature_Joint model.
            ratings_test_path: Path to the 'ratings_test.csv' file.
        """
        self.model_to_test = model_to_test
        
        # Load the test ratings
        self.R_test = pd.read_csv(ratings_test_path)
        self.R_test.replace('NA', np.nan, inplace=True)
        self.R_test = self.R_test.to_numpy()
        
        # Determine the model type
        if isinstance(model_to_test, Normal_Joint):
            self.model_type = 'Normal_Joint'
        elif isinstance(model_to_test, Feature_Joint):
            self.model_type = 'Feature_Joint'
        else:
            raise ValueError('model_to_test must be an instance of Normal_Joint or Feature_Joint')
    
    def test(self, input_size: float, top_n: int, rating_range: tuple[float, float]) -> float:
        """Test the model's accuracy or MAE on the test dataset.
        
        Args:
            input_size: Proportion of observed ratings to use as input (between 0 and 1).
            top_n: Number of top recommendations to consider.
            rating_range: Tuple representing the desired rating range.
            
        Returns:
            For Normal_Joint: Accuracy (float between 0 and 1).
            For Feature_Joint: Mean Absolute Error (MAE).
        """
        total_users = 0
        total_accuracy = 0.0  # For Normal_Joint
        total_mae = 0.0       # For Feature_Joint
        num_test_ratings = 0  # For Feature_Joint

        num_users = self.R_test.shape[0]
        for user_index in range(num_users):
            user_ratings = self.R_test[user_index, :]
            observed_indices = np.where(~np.isnan(user_ratings))[0]
            num_observed = len(observed_indices)
            
            if num_observed < 2:
                continue  # Need at least one input and one test rating
            
            num_input = max(1, int(np.floor(input_size * num_observed)))
            if num_input >= num_observed:
                num_input = num_observed - 1  # Ensure at least one test rating

            input_indices = np.random.choice(observed_indices, size=num_input, replace=False)
            testing_indices = np.setdiff1d(observed_indices, input_indices)
            
            if len(testing_indices) == 0:
                continue  # No test ratings available
            
            # Construct input ratings dictionary
            input_ratings = {int(index): user_ratings[index] for index in input_indices}
            
            # Make predictions
            try:
                predictions = self.model_to_test.predict(input_ratings, rating_range)
            except Exception:
                continue  # Skip user if prediction fails
            
            if self.model_type == 'Normal_Joint':
                # Remove already rated movies from predictions
                predictions = {k: v for k, v in predictions.items() if k not in input_ratings}
                if not predictions:
                    continue

                # Get top_n recommendations
                sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                recommended_movie_ids = [movie_id for movie_id, _ in sorted_predictions[:top_n]]
                
                # Evaluate recommendations
                num_hits = 0
                num_recommendations = 0
                for movie_id in recommended_movie_ids:
                    if movie_id in testing_indices:
                        actual_rating = user_ratings[movie_id]
                        if rating_range[0] <= actual_rating <= rating_range[1]:
                            num_hits += 1
                        num_recommendations += 1
                
                if num_recommendations > 0:
                    user_accuracy = num_hits / num_recommendations
                    total_accuracy += user_accuracy
                    total_users += 1

            elif self.model_type == 'Feature_Joint':
                # Evaluate predictions
                testing_movie_ids = [index for index in testing_indices if index in predictions]
                if not testing_movie_ids:
                    continue

                predicted_ratings = [predictions[index] for index in testing_movie_ids]
                actual_ratings = [user_ratings[index] for index in testing_movie_ids]

                if predicted_ratings:
                    user_mae = np.mean(np.abs(np.array(predicted_ratings) - np.array(actual_ratings)))
                    total_mae += user_mae * len(predicted_ratings)
                    num_test_ratings += len(predicted_ratings)
                    total_users += 1

        # Calculate final accuracy or MAE
        if self.model_type == 'Normal_Joint':
            return total_accuracy / total_users if total_users > 0 else 0.0
        elif self.model_type == 'Feature_Joint':
            return total_mae / num_test_ratings if num_test_ratings > 0 else None
        
if __name__ == "__main__":
    # Load the trained model
    U = np.random.random((10, 6)) * 1.414
    V = np.random.random((20, 6)) * 1.414
    joint = Normal_Joint(U, V)
    joint.fit()
    
    # Initialize the Tester
    tester = Tester(joint, "ratings_test.csv")
    
    # Test the model
    input_size = 0.5
    top_n = 10
    rating_range = (1, 5)
    accuracy = tester.test(input_size, top_n, rating_range)
    
    print(f"Accuracy: {accuracy}")
