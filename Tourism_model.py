import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class KNNRecommender:
    def __init__(self, attractions_path, tourism_path):
        # Load datasets
        self.attractions_df = pd.read_csv(attractions_path)
        self.tourism_df = pd.read_csv(tourism_path)

        # Merge datasets on 'id' (tourism) and 'tourist_place_id' (attractions)
        self.data = pd.merge(self.attractions_df, self.tourism_df, left_on='tourist_place_id', right_on='id')

        # Drop unnecessary columns
        self.data.drop(columns=['id', 'tourist_place_id'], inplace=True)

        # Prepare and train the model
        self.model = None
        self.preprocessor = None
        self._prepare_data()
        self._train_model()

    def _prepare_data(self):
        # Numerical features to normalize
        numeric_features = ['entry_fee', 'safety_index', 'weather_impact', 'popularity_score', 'google_reviews_sentiment']

        # Categorical features to one-hot encode
        categorical_features = ['category', 'state']

        # Preprocessing pipeline: Scale numeric, encode categorical
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )

        # Apply transformations
        self.features = self.preprocessor.fit_transform(self.data)

    def _train_model(self):
        # Train KNN model
        self.model = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.model.fit(self.features)

    def recommend(self, city, category_filter=None, budget_limit=None):
        # Filter by city
        city_data = self.data[self.data['name'] == city]

        if city_data.empty:
            return "City not found in dataset."

        # Apply category filter
        if category_filter:
            city_data = city_data[city_data['category'].str.contains(category_filter, case=False)]

        # Apply budget filter
        if budget_limit is not None:
            city_data = city_data[city_data['entry_fee'] <= budget_limit]

        if city_data.empty:
            return "No matching attractions found."

        # Get feature vector for city
        city_features = self.preprocessor.transform(city_data)

        # Find nearest neighbors
        distances, indices = self.model.kneighbors(city_features)

        # Return top-5 recommended attractions
        recommendations = city_data.iloc[indices[0]][['attraction_name', 'category', 'entry_fee', 'popularity_score']]
        return recommendations.reset_index(drop=True)

# File paths (from your uploaded files)
attractions_path = "D:/tourism_attractions.csv"
tourism_path = "D:/cleaned_tourism_db.csv"

# Initialize and train the recommender
recommender = KNNRecommender(attractions_path, tourism_path)

# Example usage
city_name = "Mumbai"
category_filter = "Cultural"
budget_limit = 300

print(f"Top attractions in {city_name} (Category: {category_filter}, Budget: {budget_limit}):\n")
print(recommender.recommend(city=city_name, category_filter=category_filter, budget_limit=budget_limit))
