import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('zomato_extracted.csv')

# Function to recommend restaurants using AI (cosine similarity)
def recommend_restaurants(current_restaurant, df, num_recommendations=3):
    # Check if the restaurant exists in the dataset
    if current_restaurant not in df['name'].values:
        return [("Current restaurant information not found, please check the restaurant name.", "", "")]

    # Create a TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the 'cuisines' column
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cuisines'])

    # Get the index of the current restaurant
    idx = df.index[df['name'] == current_restaurant].tolist()[0]

    # Compute the cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get the indices of the most similar restaurants
    similar_indices = cosine_similarities.argsort()[-num_recommendations-1:-1][::-1]

    # Prepare the recommendations
    recommendations = []
    for i in similar_indices:
        recommendations.append((df['name'].iloc[i], df['rest_type'].iloc[i], df['cuisines'].iloc[i], df['url'].iloc[i]))

    return recommendations

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

# Directly show recommendations
recommended_restaurants = recommend_restaurants(selected_restaurant, df)

# Display recommendations in a table format
if recommended_restaurants[0][0] != "Current restaurant information not found, please check the restaurant name.":
    st.subheader('Recommended Restaurants')
    
    # Creating a dataframe to display recommendations
    recommendations_df = pd.DataFrame(recommended_restaurants, columns=["Name", "Rest Type", "Cuisines", "URL"])

    # Creating clickable links for the restaurants
    recommendations_df['URL'] = recommendations_df.apply(
        lambda row: f'<a href="{row["URL"]}">{row["URL"]}</a>', axis=1
    )
    
    # Display the dataframe using st.markdown with unsafe_allow_html to render links
    st.write(recommendations_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.write(recommended_restaurants[0][0])
