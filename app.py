import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('zomato_extracted.csv')
df = df.drop_duplicates(subset='name')
df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with empty string
df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings

# Function to recommend restaurants based on cosine similarity
# Function to recommend restaurants based on cosine similarity
# Function to recommend restaurants based on cosine similarity
def recommend_restaurants(current_restaurant, df, num_recommendations=3):
    # Get the selected restaurant's cuisines
    current_cuisines = df.loc[df['name'] == current_restaurant, 'cuisines'].values
    if len(current_cuisines) == 0:
        st.write("Cuisines not found for the selected restaurant.")
        return pd.DataFrame()  # Return an empty DataFrame
    
    current_cuisines = current_cuisines[0]

    # Filter the dataframe to only include restaurants with matching cuisines
    df_filtered = df[df['cuisines'].str.contains(current_cuisines, case=False, na=False) & (df['name'] != current_restaurant)]
    
    # Handle case where no matching restaurants are found
    if df_filtered.empty:
        st.write("No similar restaurants found. Here are some random recommendations instead:")
        # Get random recommendations from the original dataframe
        random_recommendations = df.sample(n=min(num_recommendations, len(df)))
        return random_recommendations[['name', 'rest_type', 'cuisines']].drop_duplicates().sort_values(by='name')

    # Create a TF-IDF Vectorizer to analyze cuisines
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_filtered['cuisines'])
    
    # Compute cosine similarity
    sim_matrix = cosine_similarity(tfidf_matrix)
    restaurant_indices = df_filtered.index
    
    # Get the similarity scores
    sim_scores = list(enumerate(sim_matrix[0]))

    # Sort similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the recommended restaurants
    recommended_indices = [restaurant_indices[i[0]] for i in sim_scores[1:num_recommendations + 1]]

    return df.iloc[recommended_indices][['name', 'rest_type', 'cuisines']].drop_duplicates().sort_values(by='name')



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
# Get recommendations
recommended_restaurants = recommend_restaurants(selected_restaurant, df)
# Display recommendations in a table format
if not recommended_restaurants.empty:
    st.subheader('Recommended Restaurants')
    
    # Creating clickable links for the restaurants using only the restaurant name
    recommended_restaurants['Restaurant'] = recommended_restaurants.apply(
        lambda row: f'<a href="{restaurant_info["url"].values[0]}">{row["name"]}</a>', axis=1
    )
    
    # Display the dataframe using st.markdown with unsafe_allow_html to render links
    st.write(recommended_restaurants[['Restaurant', 'rest_type', 'cuisines']].to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.write("No recommendations found.")
