import pandas as pd
import numpy as np
import infer
import json
import re

class Preferences:
    def __init__(self, genres):
        self.rating_range = [3.5, 5]
        self.num_recs = 20
        self.year_range = [0, 3000]
        self.genres = genres
        self.genres_available = genres
    
    def set_preferences(self, arguments):
        pointer = 0
        while pointer < len(arguments):
            if arguments[pointer] == "--ratings_greater_than" or arguments[pointer] == "-g":
                self.rating_range[0] = float(arguments[pointer + 1])
                pointer += 2
            elif arguments[pointer] == "--ratings_less_than" or arguments[pointer] == "-l":
                self.rating_range[1] = float(arguments[pointer + 1])
                pointer += 2
            elif arguments[pointer] == "--number_recs" or arguments[pointer] == "-n":
                self.num_recs = int(arguments[pointer + 1])
                pointer += 2
            elif arguments[pointer] == "--after_year" or arguments[pointer] == "-a":
                self.year_range[0] = int(arguments[pointer + 1])
                pointer += 2
            elif arguments[pointer] == "--before_year" or arguments[pointer] == "-b":
                self.year_range[1] = int(arguments[pointer + 1])
                pointer += 2
            elif arguments[pointer] == "--genres_include" or arguments[pointer] == "-i":
                num_args = int(arguments[pointer + 1])
                pointer += 2
                for i in range(num_args):
                    self.genres.add(arguments[i + pointer])
                pointer += num_args
            elif arguments[pointer] == "--genres_exclude" or arguments[pointer] == "-e":
                num_args = int(arguments[pointer + 1])
                pointer += 2
                for i in range(num_args):
                    self.genres.remove(arguments[i + pointer])
                pointer += num_args
            elif arguments[pointer] == "--show_preferences" or arguments[pointer] == "-p":
                print(f"Rating Range: {self.rating_range}")
                print(f"Number of Recommendations: {self.num_recs}")
                print(f"Year Range: {self.year_range}")
                print(f"Genres: {self.genres}")
                print(f"Genres Available: {self.genres_available}")
                pointer += 1
            elif arguments[pointer] == "--save_preferences" or arguments[pointer] == "-s":
                self.save(arguments[pointer + 1])
                pointer += 2
            elif arguments[pointer] == "--load_preferences" or arguments[pointer] == "-d":
                self.load(arguments[pointer + 1])
                pointer += 2
            else:
                raise ValueError(arguments[pointer])
    
    def save(self, file_path: str):
        preferences = {
            "rating_range": self.rating_range,
            "num_recs": self.num_recs,
            "year_range": self.year_range,
            "genres": list(self.genres),
        }
        if not file_path.endswith(".mrp"):
            file_path = f"{file_path}.mrp"
        with open(file_path, "w") as file:
            file.write(json.dumps(preferences))

    def load(self, file_path):
        with open(file_path, "r") as file:
            preferences = json.loads(file.read())
        self.rating_range = preferences["rating_range"]
        self.num_recs = preferences["num_recs"]
        self.year_range = preferences["year_range"]
        self.genres = set(preferences["genres"])

class RecsUI:
    def __init__(self, metadata_path, model: infer.Predictor):
        self.metadata = pd.read_csv(metadata_path)

        year_pattern = r"(\(\d{4}\))"
        years = list()
        genres = set()
        for index, row in self.metadata.iterrows():
            text = row["title"]
            match = re.search(year_pattern, text)
            if match:
                year = int(match.group(1)[1:-1])
                start_index = match.start(1)
                text = text[:start_index]
                self.metadata.loc[index, "title"] = text
            else:
                year = 0
            years.append(year)

            genre = row["genres"].strip().split("|")
            genres.update(genre)
        self.metadata["year"] = years
        self.metadata.index = self.metadata["movieId"]
        self.metadata.drop("movieId", axis=1, inplace=True)

        self.model = model
        self.user_ratings = dict()
        self.preferences = Preferences(genres)

    def help(self, commands):
        help_messages = {
            "help": "  Display help information for available commands.\n"
                    "  Usage:\n"
                    "    help [command]\n",
            
            "search": "  Search for movies by keywords. Searches movie titles case-insensitively.\n"
                    "  Usage:\n"
                    "    search <keyword1> [keyword2 ...]\n",
            
            "import": "  Import user ratings from a JSON file.\n"
                    "  File should be a dictionary mapping movie IDs to ratings.\n"
                    "  Usage:\n"
                    "    import <file_path>\n",
            
            "export": "  Export current user ratings to a JSON file.\n"
                    "  Creates a .mrr (Movie Ratings Record) file.\n"
                    "  Usage:\n"
                    "    export <file_path>\n",
            
            "rate": "  Add or change ratings for specific movies.\n"
                "  Provide movie IDs and corresponding ratings in alternating order.\n"
                "  Usage:\n"
                "    rate <movie_id1> <rating1> [<movie_id2> <rating2> ...]\n",
            
            "delete": "  Delete ratings for specific movies.\n"
                    "  Provide movie IDs to remove their ratings.\n"
                    "  Usage:\n"
                    "    delete <movie_id1> [<movie_id2> ...]\n",
            
            "clear": "  Clear all user ratings.\n"
                    "  Usage:\n"
                    "    clear\n",
            
            "ratings": "  Display current user ratings.\n"
                    "  Shows rated movies and their ratings.\n"
                    "  Usage:\n"
                    "    ratings\n",
            
            "recommend": "  Generate movie recommendations based on user ratings.\n"
                        "  Uses Normal Joint, Feature Joint, and Combined recommendation models.\n"
                        "  Recommendations filtered by user preferences.\n"
                        "  Usage:\n"
                        "    recommend\n",
            
            "pref": "  Set or modify recommendation preferences.\n"
                        "  Available options:\n"
                        "    -g/--ratings_greater_than <min_rating>: Set minimum rating\n"
                        "    -l/--ratings_less_than <max_rating>: Set maximum rating\n"
                        "    -n/--number_recs <num>: Set number of recommendations\n"
                        "    -a/--after_year <year>: Set minimum year\n"
                        "    -b/--before_year <year>: Set maximum year\n"
                        "    -i/--genres_include <num_genres> <genre1> [genre2 ...]: Include genres\n"
                        "    -e/--genres_exclude <num_genres> <genre1> [genre2 ...]: Exclude genres\n"
                        "    -p/--show_preferences: Display current preferences\n"
                        "    -s/--save_preferences <file_path>: Save preferences to a .mrp (Movie Ratings Preferences) file\n"
                        "    -d/--load_preferences <file_path>: Load preferences from .mrp file\n"
                        "  Usage:\n"
                        "    pref [options]\n",
            
            "exit": "  Exit the movie recommendation system.\n"
                    "  Usage:\n"
                    "    exit\n"
        }

        if not commands:
            for command, description in help_messages.items():
                print(f"{command}\n{description}")
        else:
            for command in commands:
                if command in help_messages:
                    print(f"{command}\n{help_messages[command]}")
                else:
                    print(f"no help available for command: {command}")

    def search_movies(self, keywords):
        result_ids = list()
        for keyword in keywords:
            for index, row in self.metadata.iterrows():
                if keyword.lower() in row["title"].lower():
                    result_ids.append(index)
        results = self.metadata.iloc[result_ids]
        print(results)

    def import_ratings(self, file_path):
        with open(file_path, "r") as file:
            user_ratings = json.loads(file.read())
        self.user_ratings = dict()
        for movie_id, rating in user_ratings.items():
            self.user_ratings[int(movie_id)] = rating

    def export_ratings(self, file_path: str):
        if not file_path.endswith(".mrr"):
            file_path = f"{file_path}.mrr"
        with open(file_path, "w") as file:
            file.write(json.dumps(self.user_ratings))

    def add_ratings(self, movie_ids, ratings):
        for id, rating in zip(movie_ids, ratings):
            self.user_ratings[int(id)] = float(rating)

    def delete_ratings(self, movie_ids):
        for id in movie_ids:
            del self.user_ratings[int(id)]

    def clear_ratings(self):
        self.user_ratings = dict()

    def display_ratings(self):
        if self.user_ratings:
            ids = list(self.user_ratings.keys())
            ratings = list(self.user_ratings.values())
            rated_movies = self.metadata.iloc[ids].copy()
            rated_movies["my_ratings"] = ratings
            print(rated_movies)
        else:
            print("no user ratings")

    def __format_recs(self, recs: dict[int, tuple[float, float]]):
        rec_ids = list(recs.keys())
        rec_scores = list()
        for rec_id in rec_ids:
            rec_scores.append(recs[rec_id][0])
        movie_recs = self.metadata.iloc[rec_ids].copy()
        movie_recs["rec_scores"] = rec_scores

        exclusion_indices = list()
        for index, row in movie_recs.iterrows():
            retain = True
            if row["year"] < self.preferences.year_range[0] or row["year"] > self.preferences.year_range[1]:
                retain = False
            genres = row["genres"].strip().split("|")
            for genre in genres:
                if not (genre in self.preferences.genres):
                    retain = False
                    break
            if not retain:
                exclusion_indices.append(index)
        movie_recs.drop(exclusion_indices, inplace=True)

        movie_recs = movie_recs.sort_values(by='rec_scores', ascending=False).head(self.preferences.num_recs)
        return movie_recs

    def display_recs(self):
        recs = self.model.predict(self.user_ratings, self.preferences.rating_range)
        movie_recs = self.__format_recs(recs)
        print(movie_recs)
        
    def main_loop(self):
        while True:
            command = input("movie-recs $ ").strip().split()
            try:
                if command[0] == "help":
                    self.help(command[1:])
                elif command[0] == "search":
                    self.search_movies(command[1:])
                elif command[0] == "import":
                    self.import_ratings(command[1])
                elif command[0] == "export":
                    self.export_ratings(command[1])
                elif command[0] == "rate":
                    movie_ids = command[1::2]
                    ratings = command[2::2]
                    self.add_ratings(movie_ids, ratings)
                elif command[0] == "delete":
                    movie_ids = command[1:]
                    self.delete_ratings(movie_ids)
                elif command[0] == "clear":
                    self.clear_ratings()
                elif command[0] == "pref":
                    self.preferences.set_preferences(command[1:])
                elif command[0] == "ratings":
                    self.display_ratings()
                elif command[0] == "recommend":
                    self.display_recs()
                elif command[0] == "exit":
                    break
                else:
                    print(f"command not found: {command[0]}")
            except Exception as e:
                print(f"{command[0]}: {e}")
                self.help([command[0]])

if __name__ == "__main__":
    import infer
    U = np.load("data/U.npy")
    V = np.load("data/V.npy")
    cov_U = np.load("data/cov_U.npy")
    cov_V = np.load("data/cov_V.npy")
    predictor = infer.Predictor(U, V, cov_U, cov_V)
    interface = RecsUI("data/metadata.csv", predictor)
    interface.main_loop()

