import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random

# Load the dataset
df = pd.read_csv('zomato_extracted.csv')

df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with empty string
df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings
# Function to recommend restaurants based on cosine similarity
def recommend_restaurants(current_restaurant, df, num_recommendations=3):
    # Create a TF-IDF Vectorizer to analyze cuisines
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['cuisines'])
    
    # Compute the cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get the index of the current restaurant
    idx = df.index[df['name'] == current_restaurant].tolist()[0]
    
    # Get the pairwise similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the restaurants based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the indices of the recommended restaurants
    restaurant_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]  # exclude the first one (itself)
    
    return df.iloc[restaurant_indices][['name', 'rest_type', 'cuisines']]

# Streamlit app title
st.title('Restaurant Recommendation System')

# Restaurant selection
st.subheader('Choose a Restaurant')
restaurant_names = df['name'].tolist()
selected_restaurant = st.selectbox('Select a restaurant', restaurant_names)

# Display selected restaurant details
restaurant_info = df[df['name'] == selected_restaurant]
if not restaurant_info.empty:
    st.write(f"**Name:** {selected_restaurant}")
    st.write(f"**Restaurant Type:** {restaurant_info['rest_type'].values[0]}")
    st.write(f"**Cuisines Type:** {restaurant_info['cuisines'].values[0]}")
    st.write(f"**URL:** [{restaurant_info['name'].values[0]}]({restaurant_info['url'].values[0]})")
else:
    st.write("Restaurant not found. Please choose/enter another one!")

# Get recommendations
recommended_restaurants = recommend_restaurants(selected_restaurant, df)

# Display recommendations in a table format
if not recommended_restaurants.empty:
    st.subheader('Recommended Restaurants')
    
    # Creating clickable links for the restaurants using only the restaurant name
    recommended_restaurants['Restaurant'] = recommended_restaurants.apply(
        lambda row: f'<a href="{restaurant_info["url"].values[0]}">{row["name"]}</a>', axis=1
    )
    
    # Display the dataframe using st.markdown with unsafe_allow_html to render links
    st.write(recommended_restaurants[['Restaurant', 'rest_type', 'cuisines']].to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.write("No recommendations found.")
