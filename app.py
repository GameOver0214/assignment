import pandas as pd
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the restaurant dataset
df = pd.read_csv('zomato_extracted.csv')

# Ensure 'cuisines' column is clean
df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with empty string
df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings

# Remove duplicate restaurant names for the dropdown
unique_restaurant_names = df['name'].drop_duplicates().tolist()

# Function to recommend restaurants using TF-IDF
def recommend_restaurants(current_restaurant, df, num_recommendations=3):
    idx = df.index[df['name'] == current_restaurant].tolist()
    
    if not idx:
        return [("Current restaurant information not found, please check the restaurant name.", "", "", "")]
    
    idx = idx[0]

    # Compute TF-IDF matrix based on cuisines
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cuisines'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Get the pairwise similarity scores for the current restaurant
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the restaurants based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the most similar restaurants, excluding the current restaurant
    similar_indices = [i[0] for i in sim_scores if i[0] != idx][:num_recommendations]

    # Get the details of recommended restaurants
    if not similar_indices:
        return [("No similar restaurants found.", "", "", "")]
    
    recommended = df.iloc[similar_indices][['name', 'rest_type', 'cuisines', 'url']]
    return list(zip(recommended['name'], recommended['rest_type'], recommended['cuisines'], recommended['url']))

# Streamlit UI elements
st.title("Restaurant Recommendation System")

# Dropdown for selecting a restaurant
selected_restaurant = st.selectbox("Select a Restaurant", unique_restaurant_names)

# Display selected restaurant details
restaurant_info = df[df['name'] == selected_restaurant]
if not restaurant_info.empty:
    st.subheader("Selected Restaurant Details")
    st.write(f"**Name:** {selected_restaurant}")
    st.write(f"**Restaurant Type:** {restaurant_info['rest_type'].values[0]}")
    st.write(f"**Cuisines Type:** {restaurant_info['cuisines'].values[0]}")
    st.write(f"**URL:** [{restaurant_info['name'].values[0]}]({restaurant_info['url'].values[0]})")
else:
    st.error("Restaurant not found. Please choose/enter another one!")

# Suggest a random cuisine from the selected restaurant
def suggest_cuisine(cuisines):
    cuisines_list = [cuisine.strip() for cuisine in cuisines.split(',')]
    return random.choice(cuisines_list) if cuisines_list else "No cuisine types available"

# Display a suggested cuisine type
suggested_cuisine = suggest_cuisine(restaurant_info['cuisines'].values[0])
st.write(f"**Suggested Cuisine Type:** {suggested_cuisine}")

# Get recommendations based on TF-IDF
recommended_restaurants = recommend_restaurants(selected_restaurant, df)

# Check if recommendations are valid
if recommended_restaurants and recommended_restaurants[0][0] != "Current restaurant information not found, please check the restaurant name.":
    st.subheader('Recommended Restaurants (with similar cuisines):')
    
    recommendations_df = pd.DataFrame(recommended_restaurants, columns=["Restaurant", "Rest Type", "Cuisines", "URL"])
    
    # Remove duplicate rows
    recommendations_df = recommendations_df.drop_duplicates(subset='Restaurant')
    
    for _, row in recommendations_df.iterrows():
        st.markdown(f"[{row['Restaurant']}]({row['URL']}) - {row['Rest Type']} - {row['Cuisines']}")
else:
    st.warning("No recommendations available based on the selected restaurant and cuisine criteria.")
