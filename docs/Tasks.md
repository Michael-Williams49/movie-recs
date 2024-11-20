# Tasks

## Data Processing

Assigned to KL

### Objective

1. Fetch the MovieLens dataset.
2. Formulate the data into a training rating matrix (`ratings_train.csv`), a testing rating matrix (`ratings_test.csv`), and a metadata sheet (`metadata.csv`).
3. Quality control the data:
   - Remove users in `ratings_train.csv` with few ratings to minimize missing data and potential skew in PMF inference.
   - Ensure each dataset is below 4GB for memory management.

### Input

- [MovieLens dataset website](https://grouplens.org/datasets/movielens/)

### Output

- Paired CSV files
  - `ratings_train.csv` and `ratings_test.csv`: Rating matrices with column headers as movie IDs and row indices as user IDs. Missing values should be indicated with `NA`.
  - `metadata.csv`: Metadata with row indices as movie IDs and columns including movie title, year, genre, description, etc. Movie IDs must be consecutive integers ranging from 0 to number_of_movies - 1

### Output variations

- Multiple pairs of `ratings_train.csv`, `ratings_test.csv` and `metadata.csv` can be generated, varying in size or quality control methods.

## PMF and Rating Matrix Estimation

Assigned to CB

### Objective

Implement a PMF class with methods for fitting and validating the Probabilistic Matrix Factorization (PMF) model.

### Input

- Path to `ratings_train.csv` (from the Data Processing stage).
- Hyperparameters: D, lambda_U, lambda_V, and learning rate.

### Output

- NumPy arrays: U (user latent feature matrix) and V (movie latent feature matrix). The ith row in V must corresponds to the latent feature vector for the movie with movie ID i.
- Training error (MAE) and test error (MAE) from the validation method.

### Methods

- `__init__`
  - Takes the path to `ratings_train.csv` as input.
  - Initializes the PMF object with the rating matrix data.
- `fit`
  - Implements the PMF algorithm to learn the U and V matrices from the raw rating matrix.
  - Takes D, lambda_U, lambda_V, and learning rate as input.
  - Returns the estimated U and V matrices. The ith row in V must corresponds to the latent feature vector for the movie with movie ID i.
- `validation`
  - Takes validation_size, D, lambda_U, lambda_V, and learning rate as input.
  - Masks out a small proportion (validation_size) of observed ratings in the raw rating matrix.
  - Learns U and V from the masked rating matrix.
  - Calculates the Mean Absolute Error (MAE) between masked values in the estimated $\hat{R}$ and the raw rating matrix (test error).
  - Calculates the MAE between unmasked and observed values in the raw rating matrix and the estimated $\hat{R}$ (training error).
  - Returns a tuple containing the training error and test error.

## Joint Distribution Learning and Rating Inference

Assigned to WX

### Objective

Implement two classes for learning the joint distribution of movie ratings and performing inference for new users.

### Input

- NumPy array U (user latent feature matrix) and V (movie latent feature matrix) from the PMF stage.
- A dictionary (movie_id: new_user_rating) representing new user input.
- A tuple (lower_rating_bound, upper_rating_bound) representing the desired rating range for movie recommendations.

### Output

- Normal_Joint: A dictionary (movie_id: confidence_level) representing the confidence level for each unrated movie that the user's potential rating falls within the specified range.
- Feature_Joint: A dictionary (movie_id: estimated_rating) representing the estimated rating for each unrated movie.

### Classes and Methods

1. Normal_Joint

- `__init__`
  - Takes NumPy arrays U and V as input.
- `fit`
  - Learns a multivariate normal distribution by estimating the covariance matrix Sigma and mean vector mu of the joint distribution of movie ratings.
  - Takes no parameters and returns nothing.
- `predict`
  - Calculates the confidence level for each movie not rated by the new user, based on the conditional joint distribution given the user's input.
  - Takes the new user input dictionary and the desired rating range tuple as parameters.
  - Returns a dictionary of confidence levels for unrated movies.

2. Feature_Joint

- `__init__`
  - Takes NumPy arrays U and V as input.
- `predict`
  - Estimates the user's latent feature vector from the provided ratings and V.
  - Calculates estimated ratings for other movies using the user's latent feature vector and the V matrix.
  - Takes the new user input dictionary and the desired rating range tuple as parameters.
  - Returns a dictionary of estimated ratings for unrated movies.

## User Interface and System Tester

Assigned to YF

### User Interface

#### Objective

Develop a user interface that interacts with users and displays movie recommendations.

#### Input

- User commands and input.
- Predictions from the Joint Distribution Learning and Rating Inference stage.
- Metadata from `metadata.csv`.

#### Output

- Display of movie recommendations based on user input and preferences.

#### Functionality

- Main loop: Offers the following commands to the user:
  - Add custom rating for a movie.
  - Change rating for a movie.
  - Remove rating for a movie.
  - Clear all ratings.
  - Set recommendation rating range.
  - Set number of recommendations to display.
  - Display the user's current ratings.
  - Display recommendation.
- Display recommendation command
  - Constructs a dictionary (movie_id: user_rating) based on user input.
  - Calls the predict method from the Joint Distribution Learning and Rating Inference stage with the constructed dictionary and the user-set rating range.
  - Uses information from `metadata.csv` to display recommendations.
  - Displays the top n recommendations, where n is specified by the user.

### System Tester

#### Objective

Implement a Tester class to test the accuracy of the recommendation system.

#### Input

- model_to_test: An object from the Joint Distribution Learning and Rating Inference stage.
- Path to `ratings_test.csv` from the Data Processing stage.
- input_size: Real number between 0 and 1, representing the proportion of input ratings.
- top_n: Number of top recommendations to consider.
- rating_range: Tuple representing the desired rating range.

#### Output

- A real number between 0 and 1 representing the accuracy of the Normal_Joint model or the MAE of the Feature_Joint model.
  Methods:

#### Methods

- `__init__`
  - Takes model_to_test and the path to `ratings_test.csv` as input.
- `test`
  - Divides the ratings of each user in `ratings_test.csv` into input ratings and testing ratings based on input_size.
  - Calls the predict method from the Joint Distribution Learning and Rating Inference stage.
    - For Normal_Joint models:
      - Selects the top_n movies with the highest confidence levels as recommendations.
      - Calculates the proportion of recommended movies where the actual user rating falls within the rating_range.
    - For Feature_Joint models:
      - Computes the Mean Absolute Error (MAE) between predicted ratings and actual user ratings.
  - Returns the calculated accuracy or MAE.
