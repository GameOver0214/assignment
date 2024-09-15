import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Display the logo at the top of the app
st.image("logo.jpg", width=650)  # Adjust the width as needed
st.write("This System is built for people to get recommended restaurants based on cuisine types!")
st.write("Feel Free to get some recommended restaurants to reach your satisfaction!")

# Load the dataset
df = pd.read_csv('zomato_extracted.csv')

# Ensure 'cuisines' and 'approx_cost(for two people)' columns are clean
df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with empty string
df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings
df['approx_cost(for two people)'] = pd.to_numeric(df['approx_cost(for two people)'], errors='coerce')  # Ensure costs are numeric

# Remove duplicate restaurant names for the dropdown
unique_restaurant_names = df['name'].drop_duplicates().tolist()

# Function to recommend restaurants using TF-IDF
def recommend_restaurants(current_restaurant, df, cost_range, num_recommendations=3):
    # Get the index of the current restaurant
    idx = df.index[df['name'] == current_restaurant].tolist()
    
    if not idx:
        return [("Current restaurant information not found, please check the restaurant name.", "", "", "")]
    
    idx = idx[0]

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['cuisines'])

    # Calculate cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Get the indices of the most similar restaurants
    similar_indices = cosine_sim.argsort()[-num_recommendations-1:-1][::-1]
    
    # Prepare recommendations, filtering by cost range
    recommended = []
    for i in similar_indices:
        if df['name'][i] != current_restaurant and cost_range[0] <= df['approx_cost(for two people)'][i] <= cost_range[1]:
            recommended.append((df['name'][i], df['rate'][i], df['rest_type'][i], df['cuisines'][i], df['url'][i], df['approx_cost(for two people)'][i]))

    if not recommended:
        return [("No similar restaurants found in the selected cost range.", "", "", "")]
    
    return recommended

# Streamlit app title
st.title('Restaurant Recommendation System')

# Restaurant selection
st.subheader('Choose a Restaurant')
selected_restaurant = st.selectbox('Select a restaurant', unique_restaurant_names)

# Add a cost range slider for filtering recommendations
st.subheader('Select Average Cost for Two People')
cost_range = st.slider('Select the cost range', min_value=0, max_value=int(df['approx_cost(for two people)'].max()), value=(0, 500))

# Display selected restaurant details
restaurant_info = df[df['name'] == selected_restaurant]
if not restaurant_info.empty:
    st.write(f"**Name:** {selected_restaurant}")
    st.write(f"**Restaurant Type:** {restaurant_info['rest_type'].values[0]}")
    st.write(f"**Restaurant Rating:** {restaurant_info['rate'].values[0]}")
    st.write(f"**Cuisines Type:** {restaurant_info['cuisines'].values[0]}")
    st.write(f"**URL:** [{restaurant_info['name'].values[0]}]({restaurant_info['url'].values[0]})")
    st.write(f"**Average Cost (for Two People):** {restaurant_info['approx_cost(for two people)'].values[0]}")
else:
    st.write("Restaurant not found. Please choose/enter another one!")

# Get recommendations using TF-IDF and filter by cost range
recommended_restaurants = recommend_restaurants(selected_restaurant, df, cost_range)

# Display recommendations in a table format if valid recommendations exist
if recommended_restaurants[0][0] != "Current restaurant information not found, please check the restaurant name.":
    st.subheader('Recommended Restaurants')
    
    # Creating a dataframe to display recommendations in a table
    recommendations_df = pd.DataFrame(recommended_restaurants, columns=["Restaurant", "rate", "Rest Type", "Cuisines", "URL", "approx_cost(for two people)"])
    
    # Make restaurant names clickable links
    recommendations_df['URL'] = recommendations_df.apply(lambda x: f"[{x['Restaurant']}]({x['URL']})", axis=1)
    
    # Display the dataframe using st.markdown for clickable URLs
    st.markdown(recommendations_df.to_markdown(index=False), unsafe_allow_html=True)
else:
    st.write(recommended_restaurants[0][0])
