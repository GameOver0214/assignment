import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Display the logo at the top of the app
st.image("logo.jpg", width=650)  # Adjust the width as needed
st.write("This system is built for people to get recommended restaurants based on restaurant and cuisine types!")
st.write("Feel free to get some recommended restaurants to reach your satisfaction!")

# Load the dataset
df = pd.read_csv('zomato_extracted.csv')

# Ensure 'cuisines' column is clean
df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with an empty string
df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings

# Remove duplicate restaurant names for the dropdown
unique_restaurant_names = df['name'].drop_duplicates().tolist()

# Extract unique cuisines for the cuisine dropdown
unique_cuisines = df['cuisines'].str.split(',').explode().str.strip().drop_duplicates().tolist()

# Function to recommend restaurants using TF-IDF
def recommend_restaurants(current_restaurant, selected_cuisine, df, num_recommendations=3):
    # Filter restaurants based on the selected cuisine
    df_filtered = df[df['cuisines'].str.contains(selected_cuisine, case=False, na=False)]
    
    # Get the index of the current restaurant in the filtered dataframe
    idx = df_filtered.index[df_filtered['name'] == current_restaurant].tolist()
    
    if not idx:
        return [("Current restaurant information not found or no matching restaurants for selected cuisine.", "", "", "", "", "")]
    
    idx = idx[0]

    # TF-IDF Vectorization for cuisine types in the filtered dataframe
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_filtered['cuisines'])

    # Calculate cosine similarity between all restaurants based on cuisines
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix[idx]).flatten()

    # Get the indices of the most similar restaurants
    similar_indices = cosine_sim.argsort()[-num_recommendations-1:-1][::-1]
    
    # Prepare recommendations
    recommended = []
    for i in similar_indices:
        if df_filtered['name'][i] != current_restaurant:
            recommended.append((df_filtered['name'][i], df_filtered['rate'][i], df_filtered['rest_type'][i], df_filtered['cuisines'][i], df_filtered['url'][i], df_filtered['approx_cost(for two people)'][i]))

    if not recommended:
        return [("No similar restaurants found.", "", "", "", "", "")]
    
    return recommended

# Streamlit app title
st.title('Restaurant and Cuisine-Based Recommendation System')

# Restaurant selection
st.subheader('Choose a Restaurant')
selected_restaurant = st.selectbox('Select a restaurant', unique_restaurant_names)

# Cuisine selection
st.subheader('Choose a Cuisine')
selected_cuisine = st.selectbox('Select a cuisine', unique_cuisines)

# Display selected restaurant details
restaurant_info = df[df['name'] == selected_restaurant]
if not restaurant_info.empty:
    st.write(f"**Name:** {selected_restaurant}")
    st.write(f"**Restaurant Type:** {restaurant_info['rest_type'].values[0]}")
    st.write(f"**Rating:** {restaurant_info['rate'].values[0]}")
    st.write(f"**Cuisines Type:** {restaurant_info['cuisines'].values[0]}")
    st.write(f"**URL:** [{restaurant_info['name'].values[0]}]({restaurant_info['url'].values[0]})")
    st.write(f"**Approximate Cost (for two people):** {restaurant_info['approx_cost(for two people)'].values[0]}")
else:
    st.write("Restaurant not found. Please choose/enter another one!")

# Get recommendations using the selected restaurant and cuisine
recommended_restaurants = recommend_restaurants(selected_restaurant, selected_cuisine, df)

# Display recommendations in a table format if valid recommendations exist
if recommended_restaurants[0][0] != "Current restaurant information not found or no matching restaurants for selected cuisine.":
    st.subheader('Recommended Restaurants')
    
    # Creating a dataframe to display recommendations in a table
    recommendations_df = pd.DataFrame(recommended_restaurants, columns=["Restaurant", "rate", "Rest Type", "Cuisines", "URL", "approx_cost(for two people)"])
    
    # Make restaurant names clickable links
    recommendations_df['URL'] = recommendations_df.apply(lambda x: f"[{x['Restaurant']}]({x['URL']})", axis=1)
    
    # Display the dataframe using st.markdown for clickable URLs
    st.markdown(recommendations_df.to_markdown(index=False), unsafe_allow_html=True)
else:
    st.write(recommended_restaurants[0][0])
