# Personalized Movie Recommendation using Probabilistic Matrix Factorization and Bayesian Reasoning

## Background

Finding a movie to watch can sometimes be a challenging task. Browsing through IMDB is time consuming, while movie recommendation articles may not align well with personal tastes. Most of the movie recommendation engines available online either make suggestions based on general questions such as mood, purpose, preferred genre, and popularity, or use collaborative filtering that depends on similarities between users or movies, without probabilistic approaches that gain insight into users' appetites. Our project aims to build a system that takes into account the user's own ratings and preferences and generates recommendations based on these insights.

## Problem Setup

Our movie recommendation system takes user ratings for a small subset of movies as input and generates recommendations with the highest probability of user satisfaction.

Our movie recommendation system operates under the assumption that users' movie ratings follow a similar distribution as the training data. The joint probability distribution learned from the training data is the prior distribution. The system takes the user's ratings for a small subset of movies as input and generates a posterior distribution by conditioning the prior on the user's input. This posterior distribution reflects the updated beliefs about the user's preferences.

The system then calculates a confidence level for each movie not yet rated by the user. This confidence level represents the probability that the user's rating for a given movie falls within a user-specified range, indicating the likelihood of the user enjoying the movie. Finally, the system presents movie recommendations in descending order of confidence, prioritizing movies with the highest probability of user satisfaction.

## Training Data and Methods

We will use the [MovieLens dataset](https://grouplens.org/datasets/movielens/) for training and evaluating our recommendation system. This dataset provides user ratings for various movies, along with supplementary information such as genre, IMDB ratings, and tags. The specific MovieLens dataset will be chosen based on our computational resources.

The raw data is represented as a sparse rating matrix, $R$, with users as rows and movies as columns. To address sparsity and enable the learning of a joint probability distribution, we use Probabilistic Matrix Factorization (PMF) (Salakhutdinov, 2007). This method assumes an underlying model:

$$ R = U V^\top + \varepsilon $$

where $\varepsilon \sim N(0, \sigma^2)$ represents noise in a rating. $U$ and $V$ are latent feature matrices for users and movies, respectively. We estimate these matrices by maximizing the likelihood $P(U, V|R)$ using gradient descent

$$ U \leftarrow U - \eta (((\hat{R} - R) \times \mathbb{I}) V + \lambda_U U) $$

$$ V \leftarrow V - \eta (((\hat{R} - R) \times \mathbb{I})^\top U + \lambda_V V) $$

where $\eta$ is the learning rate, $\hat{R} = U V^\top$, $\times$ denotes element-wise multiplication, and $\mathbb{I}$ is an indicator matrix with 1 for observed ratings and 0 otherwise. The hyperparameters are $D$, $\eta$, $\lambda_U$, and $\lambda_V$, where $D$ is the number of columns in $U$ and $V$.

Using the completed rating matrix, $\hat{R}$, we learn a multivariate normal distribution over movie ratings by estimating the mean vector $\vec{\mu}$ and covariance matrix $\Sigma$ using maximum likelihood estimation.

To evaluate the system, we use a held-out test set. We compare the user-specified range with the actual user ratings for top recommendations, and calculate the proportion of actual ratings that fall within the range to assess recommendation accuracy.
