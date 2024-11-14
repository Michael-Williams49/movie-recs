import numpy as np
import pandas as pd

class PMF:
    def __init__(self, rating_path: str):
        df = pd.read_csv(rating_path)
        # Create rating matrix from observations. Row (i): userId, Col (j): movieId, Value: user i's rating og movie j.
        rating_matrix = df.pivot(index='userId', columns='movieId', values='rating')
        # Make numpy array
        self.R = rating_matrix.to_numpy()



    def fit(self, D: int, lamb_U: float, lamb_V: float, learning_rate: float) -> tuple[np.ndarray, np.ndarray, list]:

        num_epochs = 20 #TODO: change?

        num_users, num_movies = self.R.shape
        training_errors = []

        # Create mask of observed entries in R
        observed = ~np.isnan(self.R)

        # Initialize U and V matrices (normally distributed with mean 1/D)
        U = np.random.normal(scale=1./D, size=(num_users, D))
        V = np.random.normal(scale=1./D, size=(num_movies, D))

        # Train using stochastic gradient descent
        for epoch in range(num_epochs):
            for i in range(num_users):
                for j in range(num_movies):
                    if observed[i,j]:
                        prediction = np.dot(U[i,:], V[j,:]) # Predict
                        error = self.R[i, j] - prediction 

                        # Update latent factors
                        U[i, :] += learning_rate * (error * V[j, :] - lamb_U * U[i, :])
                        V[j, :] += learning_rate * (error * U[i, :] - lamb_V * V[j, :])

            # Compute the total loss
            loss = 0
            for i in range(num_users):
                for j in range(num_movies):
                    if observed[i, j]:
                        prediction = np.dot(U[i, :], V[j, :].T)
                        # TODO: Check the 0.5 is correct! (assuming sigma^2 is gone)
                        loss += 0.5 * (self.R[i, j] - prediction) ** 2 
            loss += lamb_U * np.linalg.norm(U) + lamb_V * np.linalg.norm(V)

            # Save current training error (loss)
            training_errors.append([epoch, loss])

            # Print the training progress
            print(f"Epoch: {epoch+1}, Loss: {loss}")

        return self.U, self.V, training_errors
    
    

    def validation(self, validation_size: float, D: int, lamb_U: float, lamb_V: float, learning_rate: float) -> tuple[np.ndarray, np.ndarray, list]:

        num_epochs = 20 #TODO: change?

        num_users, num_movies = self.R.shape
        errors = []
        # Create mask of observed entries in R
        observed = ~np.isnan(self.R)

        # CREATE R_TEST
        # Choose random entries
        non_nan_indices = np.argwhere(observed)
        sample_size = int(validation_size * len(non_nan_indices))
        # Get random indices
        random_indices = non_nan_indices[np.random.choice(len(non_nan_indices), sample_size, replace=False)]

        # Replaces indices with nan
        R_test = np.copy(self.R)
        R_test[tuple(random_indices.T)] = np.nan


        # Initialize U and V matrices (normally distributed with mean 1/D)
        U = np.random.normal(scale=1./D, size=(num_users, D))
        V = np.random.normal(scale=1./D, size=(num_movies, D))

        # Train using stochastic gradient descent
        for epoch in range(num_epochs):
            for i in range(num_users):
                for j in range(num_movies):
                    if observed[i,j]:
                        prediction = np.dot(U[i, :], V[j, :].T)
                        error = self.R[i, j] - prediction 

                        # Update latent factors
                        U[i, :] += learning_rate * (error * V[j, :] - lamb_U * U[i, :])
                        V[j, :] += learning_rate * (error * U[i, :] - lamb_V * V[j, :])

            # Compute the total loss
            loss = 0
            for i in range(num_users):
                for j in range(num_movies):
                    if observed[i, j]:
                        prediction = np.dot(U[i, :], V[j, :].T)
                        # TODO: Check the 0.5 is correct! (assuming sigma^2 is gone)
                        loss += 0.5 * (self.R[i, j] - prediction) ** 2 
            loss += lamb_U * np.linalg.norm(U) + lamb_V * np.linalg.norm(V)

            # Compute validation error and save
            val_err = 0
            for index in random_indices:
                val_err += np.abs(self.R[index[0], index[1]] - np.dot(U[index[0],:], V[index[1],:])) ** 2 / sample_size

            # Save current errors 
            errors.append([epoch, loss, val_err])
            # Print the training progress
            print(f"Epoch: {epoch+1}, Loss: {loss}, Validation_error: {val_err}")

        return U, V, errors

 