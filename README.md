# Movie Recommendation System using Probabilistic Matrix Factorization

## Overview
This project implements a personalized movie recommendation system using Probabilistic Matrix Factorization (PMF) and Bayesian reasoning. The system learns from users' rating patterns to suggest movies that align with individual preferences, going beyond simple genre-based or popularity-based recommendations. The model provides both predicted ratings and standard deviations for its recommendations, allowing for more informed movie suggestions. Using the MovieLens dataset (100K ratings from 600 users across 9000 movies), our system achieves 87% precision and 73% recall in recommendation accuracy. 

The recommendation system consists of the following modules:
1. `data.py`: Data preprocessing. Filter users and movies by number of ratings in raw data stored in `raw/` and create training dataset, test dataset, and metadata sheet.
2. `factorize.py`: Factorize the rating matrix into user feature matrix `U`, movie feature matrix `V`, and their corresponding covariance matrices.
3. `infer.py`: Learn the feature vector of a new user from user-provided ratings and provide recommendation including predicted ratings and standard deviations based on user-defined rating range.
4. `RecsUI.py`: The recommendation UI for interactive use.
5. `test.py`: Test the precision and recall of the recommendation system.

## Installation
The system is designed for POSIX compliant operating systems. To setup Python environment, please run

```bash
conda create -n env_name matplotlib numpy pandas scikit-learn
conda activate env_name
```

## Usage
The system can be used in 4 ways, as described below.

### Use the configured system interactively
The system has been trained and configured, with necessary files stored in `data/`. To use the system interactively, please run

```bash
python RecsUI.py
```

### Walk through the model training and testing process
The raw data have been processed with outputs stored in `data/`. To walk through the training and testing process, please run

```bash
python demo.py
```

### Use the system with your dataset
To preprocess the raw data of your dataset, please place raw data in `raw/`. Ensure that these files exist and are named correctly:
- `raw/ratings.csv`: A CSV table storing users' ratings for movies with at least these columns: `userId`, `movieId`, `rating`.
- `raw/movies.csv`: A CSV table storing metadata of movies with at least these columns: `movieId`, `title`, `genres`.

Then run

```bash
python data.py
```

You will be guided through the data preprocessing steps. The output files will be stored in `data/`.

To train and use the system with your own dataset, please ensure these files exist and correspond: `data/ratings_train.npy`, `data/metadata.csv`. Then run

```bash
python main.py
```

### Run each module independently
The modules have been configured to be able to run independently. All necessary files have been stored in `data/` and `raw/`. To run a specific module, please run

```bash
python <module_name>.py
```

## License
[![CC BY 4.0](https://mirrors.creativecommons.org/presskit/buttons/88x31/svg/by.svg)](https://creativecommons.org/licenses/by/4.0/)

This project is licensed under the Creative Commons Attribution 4.0 International License.
