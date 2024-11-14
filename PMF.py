import numpy as np
import pandas as pd

class PMF:
    def __init__(self, rating_path: str):
        df = pd.read_csv(rating_path)
        # Create rating matrix from observations. Row (i): userId, Col (j): movieId, Value: user i's rating og movie j.
        rating_matrix = df.pivot(index='userId', columns='movieId', values='rating')
        # Make numpy array
        self.R = rating_matrix.to_numpy()



    def fit(self, D: int, lamb_U: float, lamb_V: float, learning_rate: float, num_epochs: int = 1000) -> tuple[np.ndarray, np.ndarray, list]:

        num_users, num_movies = self.R.shape
        training_errors = []

        # Create mask of observed entries in R
        observed = ~np.isnan(self.R)

        # Initialize U and V matrices (normally distributed with mean 1/D)
        U = np.random.normal(scale=1./D, size=(num_users, D))
        V = np.random.normal(scale=1./D, size=(num_movies, D))

        # Create masked array for the training
        R_masked = np.ma.array(self.R, mask=~observed)
        
        for epoch in range(num_epochs):
            # Compute predictions
            predictions = np.dot(U, V.T)
            
            # Calculate errors for observed entries
            errors = np.ma.array(predictions, mask=~observed) - R_masked
            
            # Update U and V via gradient descent
            grad_U = np.dot(errors, V) + lamb_U * U
            grad_V = np.dot(errors.T, U) + lamb_V * V
            U -= learning_rate * grad_U
            V -= learning_rate * grad_V
            
            # Calculate the loss
            prediction_error = np.ma.array(predictions - self.R, mask=~observed)
            loss = 0.5 * np.sum(prediction_error**2) + lamb_U * np.sum(U**2) + lamb_V * np.sum(V**2)

            # Save errors
            training_errors.append([epoch, loss])

            # Print result of every 100 epoch
            if epoch % 100 == 0:
                print(f"Epoch: {epoch+1}, Loss: {loss}")

        return U, V, training_errors
    


    def validation(self, validation_size: float, D: int, lamb_U: float, lamb_V: float, learning_rate: float, num_epochs: int = 1000) -> tuple[np.ndarray, np.ndarray, list]:
        
        num_users, num_movies = self.R.shape
        training_errors = []

        # Create validation split
        R_observed = ~np.isnan(self.R)
        non_nan_indices = np.argwhere(R_observed)
        sample_size = int(validation_size * len(non_nan_indices))
        random_indices = non_nan_indices[np.random.choice(len(non_nan_indices), sample_size, replace=False)]
        
        # Create the training matrix (change random indices to nan)
        R_train = np.copy(self.R)
        R_train[tuple(random_indices.T)] = np.nan
        observed = ~np.isnan(R_train)
        
        # Initialize U and V matrices (normally distributed with mean 1/D)
        U = np.random.normal(scale=1./D, size=(num_users, D))
        V = np.random.normal(scale=1./D, size=(num_movies, D))
        
        # Create masked array for the training
        R_masked = np.ma.array(R_train, mask=~observed)
        
        for epoch in range(num_epochs):
            # Compute predictions
            predictions = np.dot(U, V.T)
            
            # Calculate errors for observed entries
            errors = np.ma.array(predictions, mask=~observed) - R_masked
            
            # Update U and V via gradient descent
            grad_U = np.dot(errors, V) + lamb_U * U
            grad_V = np.dot(errors.T, U) + lamb_V * V
            U -= learning_rate * grad_U
            V -= learning_rate * grad_V
            
            # Calculate the loss
            prediction_error = np.ma.array(predictions - R_train, mask=~observed)
            loss = 0.5 * np.sum(prediction_error**2) + lamb_U * np.sum(U**2) + lamb_V * np.sum(V**2)
            
            # Calculate the validation error
            val_pred = np.sum(U[random_indices[:,0]] * V[random_indices[:,1]], axis=1)
            val_true = self.R[random_indices[:,0], random_indices[:,1]]
            val_err = np.mean((val_pred - val_true)**2)
            
            # Save errors
            training_errors.append([epoch, loss, val_err])

            # Print result of every 100 epoch
            if epoch % 100 == 0:
                print(f"Epoch: {epoch+1}, Loss: {loss}, Validation_error: {val_err}")
        
        return U, V, training_errors, random_indices

    
