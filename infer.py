import numpy as np

class Predictor:
    """Implements a feature-based approach for movie rating prediction using PMF.
    
    This class uses gradient descent to find user latent features that best explain
    the observed ratings, then uses these features to predict unobserved ratings.
    
    Attributes:
        D (int): Number of latent features
        M (int): Number of movies
        U (np.ndarray): User latent feature matrix
        V (np.ndarray): Movie latent feature matrix
        learning_rate (float): Learning rate for gradient descent
        max_iter (int): Maximum number of iterations for gradient descent
        lambda_N (float): Regularization parameter
        value_limit (float): Threshold for detecting divergence
    """
    
    def __init__(self, U: np.ndarray, V: np.ndarray, cov_U: np.ndarray, cov_V: np.ndarray, learning_rate: float = 0.001, max_iter: int = 10000, lambda_N: float = 1, value_limit: float = 1e6):
        """Initialize the Predictor model.
        
        Args:
            U: User latent feature matrix
            V: Movie latent feature matrix
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations for gradient descent
            lambda_N: Regularization parameter
            value_limit: Threshold for detecting divergence in gradient descent
            
        Raises:
            AssertionError: If feature dimensions of U and V don't match
        """
        assert U.shape[1] == V.shape[1]
        self.D = U.shape[1]
        self.M = V.shape[0]
        self.U = U
        self.V = V
        self.cov_U = cov_U
        self.cov_V = cov_V
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_N = lambda_N
        self.value_limit = value_limit
    
    def __fit(self, r: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit user latent features using gradient descent.
        
        Args:
            r: Vector of ratings (0 for unknown ratings)
            i: Indicator vector (1 for observed ratings, 0 for unknown)
            
        Returns:
            Fitted user latent feature vector
            
        Raises:
            ValueError: If gradient descent diverges (learning rate too high)
        """
        # Initialize random user features
        u = np.random.random((self.D,))
        
        # Gradient descent
        for _ in range(self.max_iter):
            # Update features using gradient of regularized squared error
            u -= self.learning_rate * (((self.V @ u - r) * i).T @ self.V + self.lambda_N * u)
            
            # Check for divergence
            if np.mean(abs(u)) > self.value_limit:
                raise ValueError("learning rate is too large")
        
        H = self.V.T @ (self.V * i.reshape(-1, 1)) + self.lambda_N * np.eye(self.D)
        cov_u = np.linalg.inv(H)

        return u, cov_u
    
    def __deviation(self, u: np.ndarray, cov_u: np.ndarray, movie_id: int):
        cov_uv = np.zeros((self.D, self.D))
        v = self.V[movie_id]
        cov_v = self.cov_V[movie_id]
        for i in range(self.D):
            for j in range(self.D):
                cov_uv[i, j] = v[i] * v[j] * cov_u[i, j] + u[i] * u[j] * cov_v[i, j] + cov_u[i, j] * cov_v[i, j]

        deviation = np.sqrt(cov_uv.sum())
        return deviation
    
    def predict(self, given_ratings: dict[int, float], rating_range: tuple[float, float]) -> dict[int, tuple[float, float]]:
        """Predict ratings for unrated movies.
        
        Args:
            given_ratings: Dictionary mapping movie indices to their ratings
            rating_range: Tuple of (lower_bound, upper_bound) for desired ratings
            
        Returns:
            Dictionary mapping movie indices to predicted ratings
            
        Raises:
            AssertionError: If rating range is invalid
        """
        assert rating_range[0] < rating_range[1]
        
        # Create rating and indicator vectors
        given_indices = given_ratings.keys()
        r = np.array([(given_ratings[index] if index in given_indices else 0) for index in range(self.M)])
        i = np.array([(1 if index in given_indices else 0) for index in range(self.M)])
        
        # Fit user features and predict ratings
        u, cov_u = self.__fit(r, i)
        r_hat = self.V @ u
        
        # Return predictions for unrated movies
        predictions = dict()
        movie_ids = list(range(self.M))
        np.random.shuffle(movie_ids)
        for movie_id in movie_ids:
            if (movie_id not in given_indices) and (rating_range[0] <= r_hat[movie_id] <= rating_range[1]):
                deviation = self.__deviation(u, cov_u, movie_id)
                predictions[movie_id] = (r_hat[movie_id], deviation)
        return predictions
    
    def random_predict(self, given_ratings: dict[int, float], rating_range: tuple[float, float]) -> dict[int, tuple[float, float]]:
        predictions = self.predict(given_ratings, rating_range)
        num_recs = len(predictions)
        available_ids = [movie_id for movie_id in range(self.M) if movie_id not in list(given_ratings.keys())]
        random_ids = np.random.choice(available_ids, num_recs, replace=False)

        random_predicitons = dict()
        for movie_id in random_ids:
            random_predicitons[movie_id] = (np.random.random(), np.random.random())

        return random_predicitons


if __name__ == "__main__":
    import json

    # Test code with random matrices
    U = np.load("data/U.npy")
    V = np.load("data/V.npy")
    cov_U = np.load("data/cov_U.npy")
    cov_V = np.load("data/cov_V.npy")

    ratings = json.loads(open("data/ratings.mrr", "r").read())
    ratings = {int(movie_id): rating for movie_id, rating in ratings.items()}
    rating_range = (3.5, 5)

    # Test the predictor
    predictor = Predictor(U, V, cov_U, cov_V)
    predictions = predictor.predict(ratings, rating_range)
    print(predictions)