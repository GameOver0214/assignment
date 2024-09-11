import streamlit as st
import pandas as pd
import random
pip install tabulate

# Load the dataset
df = pd.read_csv('zomato_extracted.csv')

# Function to recommend restaurants based on the current restaurant's type
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

# Directly show recommendations without a button
recommended_restaurants = recommend_restaurants(selected_restaurant, df)

# Display recommendations in a table format if valid recommendations exist
if recommended_restaurants[0][0] != "Current restaurant information not found, please check the restaurant name.":
    st.subheader('Recommended Restaurants')
    
    # Creating a dataframe to display recommendations in a table
    recommendations_df = pd.DataFrame(recommended_restaurants, columns=["name", "url"])
    
    # Creating clickable links for the restaurants
    recommendations_df['name'] = recommendations_df.apply(lambda row: f"[{row['name']}]({row['url']})", axis=1)
    
    # Using st.markdown to display the clickable links in a table format
    st.markdown(recommendations_df[['name']].to_markdown(index=False), unsafe_allow_html=True)
else:
    st.write(recommended_restaurants[0][0])
