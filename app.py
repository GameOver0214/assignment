import streamlit as st
import pandas as pd
import random

# Sample data for 8 restaurants
data = {
    'name': [
        'Warm Oven', 'Pasta Palace', 'Sushi Central', 
        'Taco Town', 'Burger Bistro', 'Salad Station', 
        'Pizza Place', 'Dessert Delight'
    ],
    'rest_type': [
        'Bakery', 'Italian', 'Japanese', 
        'Mexican', 'American', 'Healthy', 
        'Italian', 'Dessert'
    ],
    'description': [
        'Cozy bakery with fresh pastries.',
        'Delicious pasta and Italian dishes.',
        'Fresh sushi and sashimi served daily.',
        'Tasty tacos and authentic Mexican food.',
        'Juicy burgers with gourmet toppings.',
        'Healthy salads made with fresh ingredients.',
        'Classic Italian pizzas with various toppings.',
        'Delicious desserts and sweet treats.'
    ]
}

# Create a DataFrame
df = pd.DataFrame(data)

def recommend_restaurants(current_restaurant, df, num_recommendations=3):
    # Get the type of the current restaurant
    rest_type = df[df['name'] == current_restaurant]['rest_type'].values
    
    if len(rest_type) == 0:
        return ["Current restaurant information not found, please check the restaurant name."]
    
    # Filter restaurants with the same type as the current restaurant
    same_type_restaurants = df[df['rest_type'] == rest_type[0]]
    
    # Exclude the current restaurant being viewed
    recommendations = same_type_restaurants[same_type_restaurants['name'] != current_restaurant]
    
    if recommendations.empty:
        return ["No restaurants of the same type were found."]
    
    # Randomly recommend a specified number of restaurants
    recommended_restaurants = random.sample(recommendations['name'].values.tolist(), 
                                             min(num_recommendations, len(recommendations)))
    
    return recommended_restaurants

# Streamlit app layout
st.title('Restaurant Recommendation System')

# Display restaurant information
st.subheader('Choose a Restaurant')
restaurant_names = df['name'].tolist()
selected_restaurant = st.selectbox('Select a restaurant', restaurant_names)

# Display details of the selected restaurant
restaurant_info = df[df['name'] == selected_restaurant]
st.write(f"**Name:** {selected_restaurant}")
st.write(f"**Type:** {restaurant_info['rest_type'].values[0]}")
st.write(f"**Description:** {restaurant_info['description'].values[0]}")

if st.button('Recommend Similar Restaurants'):
    # Get recommendations
    recommended_restaurants = recommend_restaurants(selected_restaurant, df)
    
    # Display recommendations
    st.subheader('Recommended Restaurants')
    for restaurant in recommended_restaurants:
        st.write(f"- {restaurant}")
