import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class PMF:
    def __init__(self, rating_path: str):
        self.R = pd.read_csv(rating_path, index_col=0)
        self.R.sort_index(axis=1, key=lambda x: x.astype(int), inplace=True)
        self.R = self.R.to_numpy()
        self.num_users, self.num_movies = self.R.shape
        self.indicators = ~np.isnan(self.R)
        self.R[~self.indicators] = 0

    def fit(self, D: int, lambda_U: float = 1, lambda_V: float = 1, learning_rate: float = 0.0001, num_epochs: int = 10000, tolerance: float = 1e-3, verbose: bool = False, verbose_step: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        # Initialize U and V matrices (normally distributed with deviation 1/D)
        U = np.random.normal(scale=1/D, size=(self.num_users, D))
        V = np.random.normal(scale=1/D, size=(self.num_movies, D))

        loss_old = 0
        training_errors = list()

        for epoch in range(num_epochs):
            # Compute predictions
            predictions = U @ V.T

            # Calculate the loss
            errors = (predictions - self.R) * self.indicators
            loss = 0.5 * np.sum(errors**2) + 0.5 * lambda_U * np.sum(U**2) + 0.5 * lambda_V * np.sum(V**2)
            MAE = np.mean(np.abs(errors[self.indicators]))

            # Print result and save errors
            if (epoch + 1) % verbose_step == 0:
                if verbose:
                    print(f"Epoch: {epoch + 1}, Loss: {loss:.4g}, MAE: {MAE:.4g}")
                else:
                    print(f"Init: {(epoch + 1) / num_epochs * 100:.0f}%", end="\r")
                training_errors.append((loss, MAE))

            # Check if converged
            if np.abs(loss_old - loss) < tolerance:
                break

            # Store old loss for next epoch
            loss_old = loss

            # Update parameters
            U -= learning_rate * (errors @ V + lambda_U * U)
            V -= learning_rate * (errors.T @ U + lambda_V * V)

        cov_U = list()
        cov_V = list()

        for i in range(self.num_users):
            H = V.T @ (V * self.indicators[i,:].reshape(-1, 1)) + lambda_U * np.eye(D)
            cov_U.append(np.linalg.inv(H))
            
        for j in range(self.num_movies):
            H = U.T @ (U * self.indicators[:,j].reshape(-1, 1)) + lambda_V * np.eye(D)
            cov_V.append(np.linalg.inv(H))

        cov_U = np.array(cov_U)
        cov_V = np.array(cov_V)
            
        return U, V, cov_U, cov_V, training_errors

    def validate(self, validation_size: float, D: int, lambda_U: float = 1, lambda_V: float = 1, learning_rate: float = 0.0001, num_epochs: int = 10000, tolerance: float = 1e-3, verbose: bool = True, verbose_step: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, np.ndarray]:
        # Create validation split
        observed_indices = np.argwhere(self.indicators)
        train_indices, val_indices = train_test_split(observed_indices, test_size=validation_size) 
        
        # Create the training matrix
        R_train = self.R.copy()
        R_train[tuple(val_indices.T)] = 0

        # Create the indicator matrices
        indicators_train = self.indicators.copy()
        indicators_train[tuple(val_indices.T)] = False

        indicators_val = self.indicators.copy()
        indicators_val[tuple(train_indices.T)] = False
        
        # Initialize U and V matrices (normally distributed with deviation 1/D)
        U = np.random.normal(scale=1/D, size=(self.num_users, D))
        V = np.random.normal(scale=1/D, size=(self.num_movies, D))
        
        loss_old = 0
        errors = list()

        for epoch in range(num_epochs):
            # Compute predictions
            predictions = U @ V.T

            # Calculate the loss
            errors_train = (predictions - R_train) * indicators_train
            loss_train = 0.5 * np.sum(errors_train**2) + 0.5 * lambda_U * np.sum(U**2) + 0.5 * lambda_V * np.sum(V**2)
            MAE_train = np.mean(np.abs(errors_train[indicators_train]))

            errors_val = (predictions - self.R) * indicators_val
            loss_val = 0.5 * np.sum(errors_val**2) + 0.5 * lambda_U * np.sum(U**2) + 0.5 * lambda_V * np.sum(V**2)
            MAE_val = np.mean(np.abs(errors_val[indicators_val]))

            # Print result and save errors
            if (epoch + 1) % verbose_step == 0:
                if verbose:
                    print(f"Epoch: {epoch + 1}\n  Training Loss: {loss_train:.4g}, Training MAE: {MAE_train:.4g}\n  Validation Loss: {loss_val:.4g}, Validation MAE: {MAE_val:.4g}")
                errors.append((loss_train, MAE_train, loss_val, MAE_val))

            # Check if converged
            if np.abs(loss_old - loss_train) < tolerance:
                break

            # Store old loss for next epoch
            loss_old = loss_train

            # Update parameters
            U -= learning_rate * (errors_train @ V + lambda_U * U)
            V -= learning_rate * (errors_train.T @ U + lambda_V * V)

        cov_U = list()
        cov_V = list()

        for i in range(self.num_users):
            H = V.T @ (V * indicators_train[i,:].reshape(-1, 1)) + lambda_U * np.eye(D)
            cov_U.append(np.linalg.inv(H))
            
        for j in range(self.num_movies):
            H = U.T @ (U * indicators_train[:,j].reshape(-1, 1)) + lambda_V * np.eye(D)
            cov_V.append(np.linalg.inv(H))

        cov_U = np.array(cov_U)
        cov_V = np.array(cov_V)
      
        return U, V, cov_U, cov_V, errors, val_indices

    
if __name__ == "__main__":
    factorization = PMF("data/ratings_train.csv")
    # factorization.validate(0.2, 10, num_epochs=3000)
    U, V, cov_U, cov_V, training_errors = factorization.fit(10, verbose=True, num_epochs=1000)
    np.save("data/U.npy", U)
    np.save("data/V.npy", V)
    np.save("data/cov_U.npy", cov_U)
    np.save("data/cov_V.npy", cov_V)