import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the restaurant dataset
df = pd.read_csv('zomato_extracted.csv')

# Ensure 'cuisines' column is clean
df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with empty string
df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings

# Remove duplicate restaurant names for the dropdown
unique_restaurant_names = df['name'].drop_duplicates().tolist()

# Function to recommend restaurants based on TF-IDF and rest_type
def recommend_restaurants(current_restaurant, df, num_recommendations=5):
    # Get the current restaurant's type and cuisines
    current_info = df[df['name'] == current_restaurant]
    if current_info.empty:
        return [("Current restaurant information not found, please check the restaurant name.", "", "", "")]
    
    current_rest_type = current_info['rest_type'].values[0]
    current_cuisines = current_info['cuisines'].values[0]

    # Filter restaurants by the same type but exclude the current restaurant
    similar_restaurants = df[(df['rest_type'] == current_rest_type) & (df['name'] != current_restaurant)]

    # Create a TF-IDF Vectorizer and fit it on the cuisines of similar restaurants
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(similar_restaurants['cuisines'])

    # Get the index of the current restaurant
    idx = similar_restaurants.index[similar_restaurants['name'] == current_restaurant].tolist()[0]

    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix).flatten()

    # Get the indices of the most similar restaurants
    sim_indices = cosine_sim.argsort()[-num_recommendations:][::-1]
    
    # Get the recommended restaurants
    recommended = similar_restaurants.iloc[sim_indices]

    # Remove duplicates from recommendations
    unique_recommendations = set()
    filtered_recommendations = []
    
    for restaurant in recommended.itertuples():
        if restaurant.name not in unique_recommendations:
            unique_recommendations.add(restaurant.name)
            filtered_recommendations.append((restaurant.name, restaurant.rest_type, restaurant.cuisines, restaurant.url))

    return filtered_recommendations

# Streamlit app title
st.title('Restaurant Recommendation System')

# Restaurant selection
st.subheader('Choose a Restaurant')
selected_restaurant = st.selectbox('Select a restaurant', unique_restaurant_names)

# Display selected restaurant details
restaurant_info = df[df['name'] == selected_restaurant]
if not restaurant_info.empty:
    st.write(f"**Name:** {selected_restaurant}")
    st.write(f"**Restaurant Type:** {restaurant_info['rest_type'].values[0]}")
    st.write(f"**Cuisines Type:** {restaurant_info['cuisines'].values[0]}")
    st.write(f"**URL:** [{restaurant_info['name'].values[0]}]({restaurant_info['url'].values[0]})")
else:
    st.write("Restaurant not found. Please choose/enter another one!")

# Suggest a cuisine type from the selected restaurant
def suggest_cuisine(cuisines):
    cuisines_list = [cuisine.strip() for cuisine in cuisines.split(',')]
    if cuisines_list:
        return random.choice(cuisines_list)
    else:
        return "No cuisine types available"

# Display a suggested cuisine type
suggested_cuisine = suggest_cuisine(restaurant_info['cuisines'].values[0])
st.write(f"**Suggested Cuisine Type:** {suggested_cuisine}")

# Get recommendations based on TF-IDF and rest_type
recommended_restaurants = recommend_restaurants(selected_restaurant, df)

# Check if recommendations are valid
if recommended_restaurants and recommended_restaurants[0][0] != "Current restaurant information not found, please check the restaurant name.":
    st.subheader('Recommended Restaurants:')
    
    # Creating a dataframe to display recommendations
    recommendations_df = pd.DataFrame(recommended_restaurants, columns=["Restaurant", "Rest Type", "Cuisines", "URL"])

    # Make restaurant names clickable links
    recommendations_df['URL'] = recommendations_df.apply(lambda x: f"[{x['Restaurant']}]({x['URL']})", axis=1)
    
    # Display the dataframe using st.markdown for clickable URLs
    st.markdown(recommendations_df.to_markdown(index=False), unsafe_allow_html=True)
else:
    st.write("No recommendations available based on the selected restaurant and cuisines.")
