import numpy as np
import pandas as pd

class PMF:
    def __init__(self, rating_path: str):
        self.R = pd.read_csv(rating_path).to_numpy()
        self.num_users, self.num_movies = self.R.shape

        # Create mask of observed entries in R
        self.observed = ~np.isnan(self.R)

        # Create masked array for training
        self.R_masked = self.R
        self.R_masked[~self.observed] = 0

    def fit(self, D: int, lamb_U: float = 0.1, lamb_V: float = 0.1, learning_rate: float = 0.001, num_epochs: int = 1000, tolerance: float = 1e-3) -> tuple[np.ndarray, np.ndarray, list]:
        # Initialize U and V matrices (normally distributed with deviation 1/D)
        U = np.random.normal(scale=1./D, size=(self.num_users, D))
        V = np.random.normal(scale=1./D, size=(self.num_movies, D))

        loss_old = 0
        training_errors = []

        for epoch in range(num_epochs):
            # Compute predictions
            predictions = np.dot(U, V.T)
            
            # Calculate errors for observed entries
            errors = predictions - self.R_masked
            
            # Update U and V via gradient descent
            grad_U = np.dot(errors, V) + lamb_U * U
            grad_V = np.dot(errors.T, U) + lamb_V * V
            U -= learning_rate * grad_U
            V -= learning_rate * grad_V
            
            # Calculate the loss
            predictions = np.dot(U, V.T)
            errors = predictions - self.R_masked
            loss = 0.5 * np.sum(errors**2) + lamb_U * np.sum(U**2) + lamb_V * np.sum(V**2)
            L1_loss = np.mean(np.abs(errors))

            # Save errors
            training_errors.append([epoch, loss, L1_loss])

            # Check if converged
            if np.abs(loss_old - loss) < tolerance:
                break

            # Store old loss for next epoch
            loss_old = loss

            # Print result of every 100 epoch
            if (epoch + 1) % 100 == 0:
                print(f"Epoch: {epoch+1}, Loss: {loss}, L1_loss: {L1_loss}")

        return U, V, training_errors


    def validation(self, validation_size: float, D: int, lamb_U: float = 0.1, lamb_V: float = 0.1, learning_rate: float = 0.001, num_epochs: int = 1000, tolerance: float = 1e-3) -> tuple[np.ndarray, np.ndarray, list]:
        # Create validation split
        observed_indices = np.argwhere(self.observed)
        val_size = int(validation_size * len(observed_indices))
        val_indices = observed_indices[np.random.choice(len(observed_indices), val_size, replace=False)]
        
        # Create the training matrix (change random indices to nan)
        R_train = np.copy(self.R)
        R_train[tuple(val_indices.T)] = np.nan
        training_observed = ~np.isnan(R_train)

        # Create masked array for the training
        R_train[~training_observed] = 0
        
        # Initialize U and V matrices (normally distributed with deviation 1/D)
        U = np.random.normal(scale=1./D, size=(self.num_users, D))
        V = np.random.normal(scale=1./D, size=(self.num_movies, D))
        
        old_loss = 0
        training_errors = []

        for epoch in range(num_epochs):
            # Compute predictions
            predictions = np.dot(U, V.T)
            
            # Calculate errors for observed entries
            errors = predictions - R_train
            
            # Update U and V via gradient descent
            grad_U = np.dot(errors, V) + lamb_U * U
            grad_V = np.dot(errors.T, U) + lamb_V * V
            U -= learning_rate * grad_U
            V -= learning_rate * grad_V
            
            # Calculate the loss
            predictions = np.dot(U, V.T)
            errors = predictions - R_train
            loss = 0.5 * np.sum(errors**2) + lamb_U * np.sum(U**2) + lamb_V * np.sum(V**2)
            L1_loss = np.mean(np.abs(errors))
            
            # Calculate the validation error
            val_pred = predictions[val_indices[:,0], val_indices[:,1]]
            val_true = self.R[val_indices[:,0], val_indices[:,1]]
            val_L1_err = np.mean(np.abs(val_pred - val_true))
            
            # Save errors
            training_errors.append([epoch, loss, L1_loss, val_L1_err])

            # Check for convergence
            if np.abs(old_loss - loss) < tolerance:
                break

            # Store old loss
            old_loss = loss

            # Print result of every 100 epoch
            if (epoch + 1) % 100 == 0:
                print(f"Epoch: {epoch+1}, Loss: {loss}, L1_loss: {L1_loss}, Validation_MAE: {val_L1_err}")
      
        return U, V, training_errors, val_indices

    
if __name__ == "__main__":
    num_users, num_items = 500, 200
    
    # Generate complete rating matrix (values between 1 and 5)
    R = 1 + 4 * np.random.random((num_users, num_items))
    
    # Randomly mask 70% of entries as missing (nan)
    mask = np.random.random((num_users, num_items)) < 0.7
    R[mask] = np.nan
    
    # Save synthetic data to CSV
    pd.DataFrame(R).to_csv("random_ratings.csv", index=False)

    factorization = PMF("random_ratings.csv")
    U, V, training_errors = factorization.fit(100, num_epochs=10000)
    U_val, V_val, val_errors, val_indices = factorization.validation(0.1, 100, num_epochs=10000)