import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# OpenWeather API Key (Replace with your actual key)
OPENWEATHER_API_KEY = "aa89631dd027d04c9ebb2000946bd7bd"

# Load and cache datasets for better performance
@st.cache_data
def load_data():
    tourism_db_path = "cleaned_tourism_db.csv"
    attractions_path = "tourism_attractions.csv"

    tourism_db = pd.read_csv(tourism_db_path)
    attractions = pd.read_csv(attractions_path)
    
    merged_df = pd.merge(tourism_db, attractions, left_on='id', right_on='tourist_place_id', how='left')
    merged_df["features"] = merged_df["state"] + " " + merged_df["name"]

    scaler = MinMaxScaler()
    merged_df[["entry_fee", "safety_index", "weather_impact", "popularity_score"]] = scaler.fit_transform(
        merged_df[["entry_fee", "safety_index", "weather_impact", "popularity_score"]]
    )

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
    text_features = tfidf_vectorizer.fit_transform(merged_df["features"])
    combined_features = np.hstack((text_features.toarray(),
                                   merged_df[["entry_fee", "safety_index", "weather_impact", "popularity_score"]].values))

    knn_model = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn_model.fit(combined_features)

    return merged_df, knn_model, tfidf_vectorizer, scaler

merged_df, knn_model, tfidf_vectorizer, scaler = load_data()

def get_destinations_by_state(state):
    return merged_df[merged_df["state"] == state]["name"].unique()

def get_city_attractions(state, destination):
    city_info = merged_df[(merged_df["state"] == state) & (merged_df["name"] == destination)].iloc[0]
    city_attractions = merged_df[merged_df["id"] == city_info["id"]]
    return city_info, city_attractions

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={OPENWEATHER_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data["main"]["temp"]
    else:
        return None

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.pinimg.com/736x/f9/cc/3a/f9cc3a2faf3cb0770d5384d1b061a216.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

footer {visibility: hidden;}

#MainMenu {visibility: hidden;}

header {visibility: hidden;}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🌍 Discover & Explore: Personalized Travel Recommendations ✈️")

state = st.selectbox("📍 Select State:", merged_df["state"].unique())
destination_options = get_destinations_by_state(state)
destination = st.selectbox("🏙️ Select City/Destination:", destination_options)

if st.button("🔍 Find Attractions"):
    city_info, attractions = get_city_attractions(state, destination)
    temperature = get_weather(destination)

    st.subheader(f"🏖️ Overview of {destination} ({state})")
    st.markdown(f"""
    - 🎉 **Main Festival:** {city_info["main_festival"]}
    - 🍽️ **Cuisine:** {city_info["cuisine"]}
    - 📅 **Best Time to Visit:** {city_info["best_time_to_visit"]}
    - 💰 **Average Entry Fee:** ₹{city_info["entry_fee"] * 83:.2f} INR
    - 🛡️ **Safety Index:** {city_info["safety_index"] * 10:.1f}/10
    - ☀️ **Weather Impact:** {city_info["weather_impact"] * 10:.1f}/10
    - 🌟 **Popularity Score:** {city_info["popularity_score"] * 10:.1f}/10
    - ✈️ **Nearest Airport:** {city_info["nearest_airport"]}
    - 🚆 **Nearest Railway Station:** {city_info["nearest_train_station"]}
    """)

    if temperature is not None:
        st.subheader("🌡️ Current Weather")
        st.markdown(f"**Temperature in {destination}:** {temperature}°C")
    else:
        st.warning("⚠️ Weather data not available. Please try again later.")

    st.subheader("📸 Attractions in This City:")
    for _, row in attractions.iterrows():
        st.markdown(f"✅ **{row['attraction_name']}**")
