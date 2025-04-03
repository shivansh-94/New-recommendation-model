# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Load datasets
tourism_db_path = "D:/cleaned_tourism_db.csv"
attractions_path = "D:/tourism_attractions.csv"

# Read datasets
tourism_db = pd.read_csv(tourism_db_path)
attractions = pd.read_csv(attractions_path)

# Merge datasets on their respective IDs
merged_df = pd.merge(tourism_db, attractions, left_on='id', right_on='tourist_place_id', how='left')

# Step 1: Data Preprocessing

# Combine relevant columns for text-based feature extraction
merged_df["features"] = (
    merged_df["state"] + " " + merged_df["name"]
)

# Normalize numeric features (entry_fee, safety_index, weather_impact, popularity_score)
scaler = MinMaxScaler()
merged_df[["entry_fee", "safety_index", "weather_impact", "popularity_score"]] = scaler.fit_transform(
    merged_df[["entry_fee", "safety_index", "weather_impact", "popularity_score"]]
)

# Apply TF-IDF vectorization on combined text features
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
text_features = tfidf_vectorizer.fit_transform(merged_df["features"])

# Combine text and numeric features
combined_features = np.hstack((
    text_features.toarray(),
    merged_df[["entry_fee", "safety_index", "weather_impact", "popularity_score"]].values
))

# Step 2: Train the KNN Model

# Initialize and train the KNN model
knn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
knn_model.fit(combined_features)

print("✅ KNN Model Successfully Trained!")

# Step 3: Recommendation Function (WITH State Restriction)

def get_recommendations(state, destination, num_recommendations=5):
    """
    Generate tourism recommendations within the same state.

    Parameters:
    - state (str): User-selected state
    - destination (str): User-selected city/destination
    - num_recommendations (int): Number of recommendations to return

    Returns:
    - DataFrame: Recommended information
    """
    
    # Create user feature for matching
    user_feature = f"{state} {destination}"

    # Vectorize user input
    user_text_features = tfidf_vectorizer.transform([user_feature]).toarray()

    # Neutral numeric values (dummy input for numeric features)
    user_numeric_features = np.array([[0.5, 0.5, 0.5, 0.5]])

    # Combine user input features
    user_combined_features = np.hstack((user_text_features, user_numeric_features))

    # Find nearest neighbors
    distances, indices = knn_model.kneighbors(user_combined_features)

    # Retrieve recommendations and restrict by state
    recommendations = merged_df.iloc[indices[0]]

    # Filter to ensure only recommendations from the SAME state
    recommendations = recommendations[recommendations["state"] == state].drop_duplicates(subset=["name"]).head(num_recommendations)

    return recommendations[["name", "state", "main_festival", "cuisine", "best_time_to_visit",
                             "attraction_name", "entry_fee", "safety_index", "weather_impact", "popularity_score"]]

# Step 4: Example Usage

# Example user input
user_state = "Maharashtra"
user_destination = "Mumbai"

# Get and display recommendations
recommendations = get_recommendations(user_state, user_destination)
print("\n🎯 Recommended Destinations (Within State Only):\n", recommendations)
