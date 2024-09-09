import pandas as pd
import random

# Streamlit app layout
st.title('Restaurant Recommendation System')

# Upload CSV file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check if required columns exist
    if 'name' in df.columns and 'rest_type' in df.columns and 'url' in df.columns:
        # Proceed with the app functionality
        st.subheader('Choose a Restaurant')
        restaurant_names = df['name'].tolist()
        selected_restaurant = st.selectbox('Select a restaurant', restaurant_names)

        # Function to recommend restaurants
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

        # Display details of the selected restaurant
        restaurant_info = df[df['name'] == selected_restaurant]
        if not restaurant_info.empty:
            st.write(f"**Name:** {selected_restaurant}")
            st.write(f"**Type:** {restaurant_info['rest_type'].values[0]}")
            st.write(f"**URL:** {restaurant_info['url'].values[0]}")

            # Automatically recommend similar restaurants based on the selected restaurant
            recommended_restaurants = recommend_restaurants(selected_restaurant, df)

            # Display recommendations
            st.subheader('Recommended Restaurants')
            if recommended_restaurants:
                for restaurant, url in recommended_restaurants:
                    st.write(f"- [{restaurant}]({url})")
            else:
                st.write("No recommendations available.")
        else:
            st.write("Restaurant not found.")
    else:
        st.error("The uploaded CSV does not have the required columns: 'name', 'rest_type', and 'url'.")
