import streamlit as st
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=90)

# Create and train a model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Load the dataset
df = pd.read_csv('zomato_extracted.csv')

# Convert 'rating' to numeric, forcing errors to NaN
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Ensure 'cuisines' column is clean
df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with empty string
df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings

# Remove duplicate restaurant names for the dropdown
unique_restaurant_names = df['name'].drop_duplicates().tolist()

# Function to recommend restaurants based on cuisines and ratings
def recommend_restaurants(current_restaurant, df, num_recommendations=3):
    # Get the index of the current restaurant
    idx = df.index[df['name'] == current_restaurant].tolist()
    
    if not idx:
        return [("Current restaurant information not found, please check the restaurant name.", "", "", "")]
    
    idx = idx[0]

    # Get the cuisines of the selected restaurant
    current_cuisines = df['cuisines'][idx]
    cuisines_list = [cuisine.strip() for cuisine in current_cuisines.split(',')]

    if not cuisines_list:
        return [("No cuisines information available for the selected restaurant.", "", "", "")]

    # Filter restaurants based on any of the cuisines of the selected restaurant and rating > 4
    similar_restaurants = df[(df['cuisines'].apply(lambda x: any(cuisine in x for cuisine in cuisines_list))) & (df['rate'] > 4)]
    
    # Exclude the current restaurant from recommendations
    similar_restaurants = similar_restaurants[similar_restaurants['name'] != current_restaurant]

    # If there are no similar restaurants, return an appropriate message
    if similar_restaurants.empty:
        return [("No similar restaurants found based on cuisines and ratings.", "", "", "")]

    # Limit the number of recommendations
    recommended = similar_restaurants.sample(n=min(num_recommendations, similar_restaurants.shape[0]))

    return list(zip(recommended['name'], recommended['rest_type'], recommended['cuisines'], recommended['url']))

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

# Get recommendations based on cuisines and ratings > 4
recommended_restaurants = recommend_restaurants(selected_restaurant, df)

# Check if recommendations are valid
if recommended_restaurants and recommended_restaurants[0][0] != "Current restaurant information not found, please check the restaurant name.":
    st.subheader('Recommended Restaurants (with Rating > 4)')
    
    # Creating a dataframe to display recommendations in a table
    recommendations_df = pd.DataFrame(recommended_restaurants, columns=["Restaurant", "Rest Type", "Cuisines", "URL"])

    # Remove duplicate rows
    recommendations_df = recommendations_df.drop_duplicates()
    
    # Make restaurant names clickable links
    recommendations_df['URL'] = recommendations_df.apply(lambda x: f"[{x['Restaurant']}]({x['URL']})", axis=1)
    
    # Display the dataframe using st.markdown for clickable URLs
    st.markdown(recommendations_df.to_markdown(index=False), unsafe_allow_html=True)
else:
    st.write("No recommendations available based on the selected restaurant and rating criteria.")
