# First, I'm going to import the libraries and modules I need for this project.
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Here, I specify the paths to the datasets I'll be working with.
BOOKS_PATH = "C:/Users/98938/Desktop/Desktop/Book Recommendation/Books.csv"
USERS_PATH = "C:/Users/98938/Desktop/Desktop/Book Recommendation/Users.csv"
RATINGS_PATH = "C:/Users/98938/Desktop/Desktop/Book Recommendation/Ratings.csv"

class DataProcessor:
    def __init__(self, books_path, users_path, ratings_path):
        # I'm setting up paths to my datasets here and calling a method to load them.
        self.books_path = books_path
        self.users_path = users_path
        self.ratings_path = ratings_path
        self._load_data()

    def _load_data(self):
        # Let's load the data from CSV files into pandas DataFrames.
        self.books = pd.read_csv(self.books_path, low_memory=False)
        self.users = pd.read_csv(self.users_path)
        self.ratings = pd.read_csv(self.ratings_path)
        self._merge_data()

    def _merge_data(self):
        # Now, I'll merge the datasets together so I can work with a single dataset.
        self.dataset = pd.merge(self.books, self.ratings, on='ISBN', how='inner')
        self.dataset = pd.merge(self.dataset, self.users, on='User-ID', how='inner')
        self._clean_data()

    def _clean_data(self):
        # I need to clean the data to remove duplicates and handle missing values.
        self.dataset.drop_duplicates(inplace=True)
        self.dataset.dropna(inplace=True)
        self.dataset['User-ID'] = self.dataset['User-ID'].astype(int)
        self._preprocess_data()

    def _preprocess_data(self):
        # Preprocessing is crucial. I'm filtering out unrated books here.
        self.dataset1 = self.dataset[self.dataset['Book-Rating'] != 0]
        self.dataset1 = self.dataset1.reset_index(drop=True)

    def limit_data(self, N):
        # To make the dataset manageable, I'm limiting it to the top N users and books.
        top_users = self.dataset1['User-ID'].value_counts().index[:N]
        top_books = self.dataset1['Book-Title'].value_counts().index[:N]
        self.dataset1 = self.dataset1[self.dataset1['User-ID'].isin(top_users) &
                                      self.dataset1['Book-Title'].isin(top_books)]
        return self.dataset1

class Recommender:
    def __init__(self, data):
        # Here's where I set up the recommender system with the processed data.
        self.data = data
        self._init_model()

    def _init_model(self):
        # I'm initializing my model here. TF-IDF and Nearest Neighbors to start.
        self.tf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')
        self.tfidf_matrix = self.tf.fit_transform(self.data['Book-Title'])
        self.matrix = self.data.pivot_table(index='Book-Title', columns='User-ID',
                                            values='Book-Rating').fillna(0)
        self.model = NearestNeighbors(metric='cosine', algorithm='brute')
        self.model.fit(self.matrix)

    def _recommendations(self, book_title):
        # Combining content-based and collaborative filtering for recommendations.
        similar_items = self._content_based_recommendations(book_title, 2)
        collaborative_items = self._collaborative_recommendations(book_title, 2)
        return similar_items + collaborative_items

    def _content_based_recommendations(self, book_title, number_of_recommendations):
        # Here, I find books similar to the one provided by the user.
        idx = self.data.index[self.data['Book-Title'] == book_title].tolist()[0]
        tfidf_vector = self.tfidf_matrix[idx]
        cosine_similarities = cosine_similarity(self.tfidf_matrix, tfidf_vector.reshape(1, -1))
        similar_indices = cosine_similarities.argsort()[::-1]
        similar_items = [self.data['Book-Title'][i]
                         for i in similar_indices.flatten()
                         if self.data['Book-Title'][i] != book_title][:number_of_recommendations]
        return similar_items

    def _collaborative_recommendations(self, book_title, number_of_recommendations):
        # And here, I look for books that users with similar tastes have enjoyed.
        distances, indices = self.model.kneighbors(self.matrix.loc[book_title].values.reshape(1, -1),
                                                   n_neighbors=number_of_recommendations)
        collaborative_items = [self.matrix.index[i] for i in indices.flatten()
                               if self.matrix.index[i] != book_title]
        return collaborative_items

def main():
    # This is where everything comes together. Let's make some recommendations!
    processor = DataProcessor(BOOKS_PATH, USERS_PATH, RATINGS_PATH)
    limited_data = processor.limit_data(10000)
    recommender = Recommender(limited_data)
    book_title = input("Enter a Book Title: ")
    recommendations = recommender._recommendations(book_title)
    if recommendations:
        print("Recommended Books For You:")
        for book in recommendations:
            print(book)

if __name__ == "__main__":
    # If this script is run as the main program, let's go ahead and call main().
    main()
