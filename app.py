import streamlit as st
import pandas as pd
import random

df = pd.read_csv('zomato_extracted.csv')

def recommend_restaurants(current_restaurant, df, num_recommendations=3):
    # Get the type of the current restaurant
    rest_type = df[df['name'] == current_restaurant]['rest_type'].values
    
    if len(rest_type) == 0:
        return [("Current restaurant information not found, please check the restaurant name.", "")]
    
    # Filter restaurants with the same type as the current restaurant
    same_type_restaurants = df[df['rest_type'] == rest_type[0]]
    
    # Exclude the current restaurant being viewed
    recommendations = same_type_restaurants[same_type_restaurants['name'] != current_restaurant]
    
    if recommendations.empty:
        return [("No restaurants of the same type were found.", "")]
    
    # Randomly recommend a specified number of restaurants
    recommended = random.sample(recommendations.to_dict(orient='records'), 
                                 min(num_recommendations, len(recommendations)))
    
    return [(rest['name'], rest['url']) for rest in recommended]

st.title('Restaurant Recommendation System')

st.subheader('Choose a Restaurant')
restaurant_names = df['name'].tolist()
selected_restaurant = st.selectbox('Select a restaurant', restaurant_names)

restaurant_info = df[df['name'] == selected_restaurant]
if not restaurant_info.empty:
    st.write(f"**Name:** {selected_restaurant}")
    st.write(f"**Restaurant Type:** {restaurant_info['rest_type'].values[0]}")
    st.write(f"**Cuisines Type:** {restaurant_info['cuisines'].values[0]}")
    st.write(f"**URL:**[{restaurant_info['name'].values[0]}]({restaurant_info['url'].values[0]})")

   
else:
    st.write("Restaurant not found. Please Choose / Enter another one!")

if st.button('Recommend Similar Restaurants'):
    # Get recommendations
    recommended_restaurants = recommend_restaurants(selected_restaurant, df)
    
    # Display recommendations
    st.subheader('Recommended Restaurants')
    for restaurant, url in recommended_restaurants:
        st.write(f"- [{restaurant}]({url})")
