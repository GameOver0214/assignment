import streamlit as st
import pandas as pd

# Load the dataset
df = pd.read_csv('zomato_extracted.csv')

# Ensure 'cuisines' column is clean
df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with empty string
df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings

# Remove duplicate restaurant names for the dropdown
unique_restaurant_names = df['name'].drop_duplicates().tolist()

# Function to recommend restaurants based on the current restaurant's type
def recommend_restaurants(current_restaurant, df, num_recommendations=3):
    # Get the type of the current restaurant
    rest_type = df[df['name'] == current_restaurant]['rest_type'].values
    
    if len(rest_type) == 0:
        return [("Current restaurant information not found, please check the restaurant name.", "", "")]
    
    # Filter restaurants with the same type as the current restaurant
    same_type_restaurants = df[df['rest_type'] == rest_type[0]]
    
    # Exclude the current restaurant being viewed
    recommendations = same_type_restaurants[same_type_restaurants['name'] != current_restaurant]
    
    if recommendations.empty:
        return [("No restaurants of the same type were found.", "", "")]
    
    # Randomly recommend a specified number of restaurants, but ensure the number does not exceed available restaurants
    num_recommendations = min(num_recommendations, len(recommendations))
    recommended = recommendations.sample(n=num_recommendations)

    return [(rest['name'], rest['rest_type'], rest['cuisines'], rest['url']) for _, rest in recommended.iterrows()]

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

# Directly show recommendations without a button
recommended_restaurants = recommend_restaurants(selected_restaurant, df)

# Display recommendations in a table format if valid recommendations exist
if recommended_restaurants[0][0] != "Current restaurant information not found, please check the restaurant name.":
    st.subheader('Recommended Restaurants')
    
    # Creating a dataframe to display recommendations in a table
    recommendations_df = pd.DataFrame(recommended_restaurants, columns=["Restaurant", "Rest Type", "Cuisines", "URL"])
    
    # Make URLs clickable
    recommendations_df['URL'] = recommendations_df.apply(lambda x: f"[{x['Restaurant']}]({x['URL']})", axis=1)
    
    # Display the dataframe using st.markdown for clickable URLs
    st.markdown(recommendations_df.to_markdown(index=False), unsafe_allow_html=True)
else:
    st.write(recommended_restaurants[0][0])
