import numpy as np
import scipy.stats as stats

class Normal_Joint:
    def __init__(self, U: np.ndarray, V: np.ndarray):
        assert U.shape[1] == V.shape[1]
        self.U = U
        self.V = V
        self.R = U @ V.T
        self.rank = np.linalg.matrix_rank(self.R)

    def _conditional(self, given_ratings: dict[int, float]) -> tuple[list, np.ndarray, np.ndarray]:
        given_indices = list(given_ratings.keys())
        other_indices = [i for i in range(len(self.joint_mean)) if i not in given_indices]

        other_mean = self.joint_mean[other_indices]
        given_mean = self.joint_mean[given_indices]
        given_values = np.array(list(given_ratings.values()))
        
        other_other_cov = self.joint_cov[np.ix_(other_indices, other_indices)]
        other_given_cov = self.joint_cov[np.ix_(other_indices, given_indices)]
        given_other_cov = self.joint_cov[np.ix_(given_indices, other_indices)]
        given_given_cov = self.joint_cov[np.ix_(given_indices, given_indices)]
        
        conditional_mean = other_mean + other_given_cov @ np.linalg.inv(given_given_cov) @ (given_values - given_mean)
        conditional_cov = other_other_cov - other_given_cov @ np.linalg.inv(given_given_cov) @ given_other_cov

        return other_indices, conditional_mean, conditional_cov
    
    def _probability(self, mean: float, variance: float, lower_bound: float, upper_bound: float) -> float:
        std_dev = np.sqrt(variance)
        distribution = stats.norm(loc=mean, scale=std_dev)
        probability = distribution.cdf(upper_bound) - distribution.cdf(lower_bound)
        return probability

    def fit(self):
        self.joint_mean = np.mean(self.R, axis=0)
        self.joint_cov = np.cov(self.R, rowvar=False)
    
    def predict(self, given_ratings: dict[int, float], rating_range: tuple[float, float]) -> dict[int, float]:
        assert rating_range[0] < rating_range[1]
        assert len(given_ratings) < self.rank
        conditional_ids, conditional_mean, conditional_cov = self._conditional(given_ratings)
        predictions = dict()
        for index, id in enumerate(conditional_ids):
            predictions[id] = self._probability(conditional_mean[index], conditional_cov[index][index], rating_range[0], rating_range[1])
        return predictions
    
class Feature_Joint:
    def __init__(self, U: np.ndarray, V: np.ndarray, learning_rate: float = 0.001, max_iter: int = 10000, lambda_N: float = 1, value_limit: float = 1e6):
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
        u = np.random.random((self.D,))
        for _ in range(self.max_iter):
            u -= self.learning_rate * (((self.V @ u - r) * i).T @ self.V + self.lambda_N * u)
            if np.mean(abs(u)) > self.value_limit:
                raise ValueError("learning rate is too large")
        return u

    def predict(self, given_ratings: dict[int, float], rating_range: tuple[float, float]) -> dict[int, float]:
        assert rating_range[0] < rating_range[1]
        given_indices = given_ratings.keys()
        r = np.array([(given_ratings[index] if index in given_indices else 0) for index in range(self.M)])
        i = np.array([(1 if index in given_indices else 0) for index in range(self.M)])
        u = self._fit(r, i)
        r_hat = self.V @ u
        prediction = dict()
        for index in range(self.M):
            if index not in given_indices:
                prediction[index] = r_hat[index]
        return prediction


if __name__ == "__main__":
    U = np.random.random((10, 6)) * 1.414
    V = np.random.random((20, 6)) * 1.414

    given_ratings = {np.random.randint(0, 19): np.random.random() * 5 for _ in range(5)}
    rating_range = (4, 5)

    joint = Normal_Joint(U, V)
    joint.fit()
    predictions = joint.predict(given_ratings, rating_range)
    print(predictions)

    joint = Feature_Joint(U, V)
    predictions = joint.predict(given_ratings, rating_range)
    print(predictions)

