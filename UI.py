import pandas as pd
import numpy as np
from infer import Normal_Joint, Feature_Joint

class RecommendationUI:
    def __init__(self, metadata_path):
        self.metadata = pd.read_csv(metadata_path)
        self.user_ratings = {}
        self.rating_range = (0, 5)
        self.num_recommendations = 10

    def add_rating(self, movie_id, rating):
        self.user_ratings[movie_id] = rating

    def update_rating(self, movie_id, rating):
        if movie_id in self.user_ratings:
            self.user_ratings[movie_id] = rating

    def delete_rating(self, movie_id):
        if movie_id in self.user_ratings:
            del self.user_ratings[movie_id]

    def clear_rating(self):
        self.user_ratings = {}

    def set_preferences(self, rating_range, num_recommendations):
        self.rating_range = rating_range
        self.num_recommendations = num_recommendations

    def display_ratings(self):
        print("Current Ratings:")
        for movie_id, rating in self.user_ratings.items():
            movie_info = self.metadata[self.metadata['movieId'] == movie_id]
            print(f"Movie: {movie_info['title'].values[0]} | Rating: {rating}")

    def display_recommendations(self, normal_model, feature_model):
        normal_recommendations = normal_model.predict(self.user_ratings, self.rating_range)
        feature_recommendations = feature_model.predict(self.user_ratings, self.rating_range)
        combined_recommendations = dict()
        for movie_id in normal_recommendations:
            if movie_id in feature_recommendations:
                combined_recommendations[movie_id] = normal_recommendations[movie_id] * feature_recommendations[movie_id]

        sorted_normal_recommendations = sorted(normal_recommendations.items(), key=lambda x: x[1], reverse=True)[:self.num_recommendations]
        sorted_feature_recommendations = sorted(feature_recommendations.items(), key=lambda x: x[1], reverse=True)[:self.num_recommendations]
        sorted_combined_recommendations = sorted(combined_recommendations.items(), key=lambda x: x[1], reverse=True)[:self.num_recommendations]

        print("Normal Joint Recommendations:")
        for movie_id, score in sorted_normal_recommendations:
            movie_info = self.metadata[self.metadata['movieId'] == movie_id]
            print(f"Movie: {movie_info['title'].values[0]} | Score: {score}")

        print("Feature Joint Recommendations:")
        for movie_id, score in sorted_feature_recommendations:
            movie_info = self.metadata[self.metadata['movieId'] == movie_id]
            print(f"Movie: {movie_info['title'].values[0]} | Score: {score}")

        print("Combined Recommendations:")
        for movie_id, score in sorted_combined_recommendations:
            movie_info = self.metadata[self.metadata['movieId'] == movie_id]
            print(f"Movie: {movie_info['title'].values[0]} | Score: {score}")

    def main_loop(self, normal_model, feature_model):
        while True:
            print("\nCommands: add, update, delete, preferences, display_ratings, recommend, exit")
            command = input("Enter command: ").strip().lower()

            if command == "add":
                try:
                    movie_id = int(input("Enter movie ID: "))
                    rating = float(input("Enter rating: "))
                    self.add_rating(movie_id, rating)
                except ValueError:
                    print("Invalid input. Please enter a valid movie ID and rating.")
            elif command == "update":
                try:
                    movie_id = int(input("Enter movie ID: "))
                    rating = float(input("Enter new rating: "))
                    self.update_rating(movie_id, rating)
                except ValueError:
                    print("Invalid input. Please enter a valid movie ID and rating.")
            elif command == "delete":
                try:
                    movie_id = int(input("Enter movie ID: "))
                    self.delete_rating(movie_id)
                except ValueError:
                    print("Invalid input. Please enter a valid movie ID.")
            elif command == "preferences":
                try:
                    lower = float(input("Enter lower rating bound: "))
                    upper = float(input("Enter upper rating bound: "))
                    num = int(input("Enter number of recommendations: "))
                    self.set_preferences((lower, upper), num)
                except ValueError:
                    print("Invalid input. Please enter valid bounds and number of recommendations.")
            elif command == "display_ratings":
                self.display_ratings()
            elif command == "recommend":
                self.display_recommendations(normal_model, feature_model)
            elif command == "exit":
                break
            else:
                print("Invalid command.")

if __name__ == "__main__":
    # Test code with random matrices
    np.random.seed(42)
    U = np.random.random((10, 6)) * 1.414
    V = np.random.random((20, 6)) * 1.414
    joint = Normal_Joint(U, V)
    joint.fit()
    UI = RecommendationUI(U, V, None)
    # UI = RecommendationUI(U, V, 'metadata.csv')
    UI.main_loop(joint)

