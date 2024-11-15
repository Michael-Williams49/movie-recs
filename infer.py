import numpy as np
import scipy.stats as stats

class Normal_Joint:
    """Implements a joint normal distribution for movie ratings using PMF.
    
    This class creates a multivariate normal distribution from user-movie rating matrices
    obtained through Probabilistic Matrix Factorization (PMF). It provides methods to
    compute conditional probabilities for predicting user ratings.
    
    Attributes:
        U (np.ndarray): User latent feature matrix of shape (n_users, n_features)
        V (np.ndarray): Movie latent feature matrix of shape (n_movies, n_features)
        R (np.ndarray): Complete rating matrix computed as U @ V.T
        rank (int): Rank of the rating matrix R
        joint_mean (np.ndarray): Mean vector of the joint distribution
        joint_cov (np.ndarray): Covariance matrix of the joint distribution
    """
    
    def __init__(self, U: np.ndarray, V: np.ndarray):
        """Initialize the Normal_Joint model.
        
        Args:
            U: User latent feature matrix from PMF
            V: Movie latent feature matrix from PMF
            
        Raises:
            AssertionError: If the feature dimensions of U and V don't match
        """
        assert U.shape[1] == V.shape[1]
        self.U = U
        self.V = V
        self.R = U @ V.T
        self.rank = np.linalg.matrix_rank(self.R)

    def _conditional(self, given_ratings: dict[int, float]) -> tuple[list, np.ndarray, np.ndarray]:
        """Compute the conditional distribution given observed ratings.
        
        Uses the properties of multivariate normal distribution to compute
        the conditional distribution parameters.
        
        Args:
            given_ratings: Dictionary mapping movie indices to their ratings
            
        Returns:
            tuple containing:
                - list of indices for movies without ratings
                - conditional mean vector
                - conditional covariance matrix
        """
        # Extract indices for given and unknown ratings
        given_indices = list(given_ratings.keys())
        other_indices = [i for i in range(len(self.joint_mean)) if i not in given_indices]

        # Split mean vector into given and unknown components
        other_mean = self.joint_mean[other_indices]
        given_mean = self.joint_mean[given_indices]
        given_values = np.array(list(given_ratings.values()))
        
        # Partition covariance matrix
        other_other_cov = self.joint_cov[np.ix_(other_indices, other_indices)]
        other_given_cov = self.joint_cov[np.ix_(other_indices, given_indices)]
        given_other_cov = self.joint_cov[np.ix_(given_indices, other_indices)]
        given_given_cov = self.joint_cov[np.ix_(given_indices, given_indices)]
        
        # Compute conditional distribution parameters
        conditional_mean = other_mean + other_given_cov @ np.linalg.inv(given_given_cov) @ (given_values - given_mean)
        conditional_cov = other_other_cov - other_given_cov @ np.linalg.inv(given_given_cov) @ given_other_cov

        return other_indices, conditional_mean, conditional_cov
    
    def _probability(self, mean: float, variance: float, lower_bound: float, upper_bound: float) -> float:
        """Calculate probability that a rating falls within a specified range.
        
        Args:
            mean: Mean of the normal distribution
            variance: Variance of the normal distribution
            lower_bound: Lower bound of the desired rating range
            upper_bound: Upper bound of the desired rating range
            
        Returns:
            Probability that the rating falls within [lower_bound, upper_bound]
        """
        std_dev = np.sqrt(variance)
        distribution = stats.norm(loc=mean, scale=std_dev)
        probability = distribution.cdf(upper_bound) - distribution.cdf(lower_bound)
        return probability

    def fit(self):
        """Fit the joint normal distribution to the rating matrix.
        
        Computes the mean vector and covariance matrix of the joint distribution
        using the complete rating matrix R.
        """
        self.joint_mean = np.mean(self.R, axis=0)
        self.joint_cov = np.cov(self.R, rowvar=False)
    
    def predict(self, given_ratings: dict[int, float], rating_range: tuple[float, float]) -> dict[int, float]:
        """Predict probabilities of movies being rated within a specified range.
        
        Args:
            given_ratings: Dictionary mapping movie indices to their ratings
            rating_range: Tuple of (lower_bound, upper_bound) for desired ratings
            
        Returns:
            Dictionary mapping movie indices to their probability of being rated
            within the specified range
            
        Raises:
            AssertionError: If rating range is invalid or too many ratings are given
        """
        assert rating_range[0] < rating_range[1]
        assert len(given_ratings) < self.rank
        conditional_ids, conditional_mean, conditional_cov = self._conditional(given_ratings)
        predictions = dict()
        for index, id in enumerate(conditional_ids):
            predictions[id] = self._probability(
                conditional_mean[index],
                conditional_cov[index][index],
                rating_range[0],
                rating_range[1]
            )
        return predictions
    
class Feature_Joint:
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
    
    def __init__(self, U: np.ndarray, V: np.ndarray, learning_rate: float = 0.001,
                 max_iter: int = 10000, lambda_N: float = 1, value_limit: float = 1e6):
        """Initialize the Feature_Joint model.
        
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
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_N = lambda_N
        self.value_limit = value_limit
    
    def _fit(self, r: np.ndarray, i: np.ndarray) -> np.ndarray:
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
        return u

    def predict(self, given_ratings: dict[int, float], rating_range: tuple[float, float]) -> dict[int, float]:
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
        u = self._fit(r, i)
        r_hat = self.V @ u
        
        # Return predictions for unrated movies
        prediction = dict()
        for index in range(self.M):
            if index not in given_indices:
                prediction[index] = r_hat[index]
        return prediction


if __name__ == "__main__":
    # Test code with random matrices
    U = np.random.random((10, 6)) * 1.414
    V = np.random.random((20, 6)) * 1.414

    # Generate random ratings for testing
    given_ratings = {np.random.randint(0, 19): np.random.random() * 5 for _ in range(5)}
    rating_range = (4, 5)

    # Test Normal_Joint
    joint = Normal_Joint(U, V)
    joint.fit()
    predictions = joint.predict(given_ratings, rating_range)
    print(predictions)

    # Test Feature_Joint
    joint = Feature_Joint(U, V)
    predictions = joint.predict(given_ratings, rating_range)
    print(predictions)