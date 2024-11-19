import pandas as pd
import numpy as np
from infer import Normal_Joint, Feature_Joint

class RecommendationUI:
    def __init__(self, U, V, metadata_path=None):
        self.U = U
        self.V = V
        self.metadata = pd.DataFrame() if metadata_path==None else pd.read_csv(metadata_path)
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
        if self.metadata.empty:
            for movie_id, rating in self.user_ratings.items():
                print(f"Movie ID: {movie_id} | Rating: {rating}")
        else:
            for movie_id, rating in self.user_ratings.items():
                movie_info = self.metadata.loc[self.metadata['movie_id'] == movie_id]
                print(f"Movie: {movie_info['title'].values[0]} | Rating: {rating}")

    def display_recommendations(self, model):
        if isinstance(model, Normal_Joint):
            recommendations = model.predict(self.user_ratings, self.rating_range)
        elif isinstance(model, Feature_Joint):
            recommendations = model.predict(self.user_ratings, self.rating_range)

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:self.num_recommendations]
        print("Recommendations:")
        for movie_id, score in sorted_recommendations:
            if self.metadata.empty:
                print(f"Movie ID: {movie_id} | Score: {score}")
            else:
                movie_info = self.metadata.loc[self.metadata['movie_id'] == movie_id]
                print(f"Movie: {movie_info['title'].values[0]} | Score: {score}")

    def main_loop(self, model):
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
                self.display_recommendations(model)
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

