import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('zomato_extracted.csv')

# Ensure 'cuisines' column is clean
df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with empty string
df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings

# Encode categorical features (cuisines)
df_encoded = pd.get_dummies(df, columns=['cuisines'], prefix='cuisine')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded.drop(columns=['name', 'rest_type', 'url']))

# Fit KNN model
knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(X_scaled)

# Streamlit app title
st.title('Restaurant Recommendation System')

# Restaurant selection
st.subheader('Choose a Restaurant')
selected_restaurant = st.selectbox('Select a restaurant', df['name'].unique())

# Get the index of the selected restaurant
selected_index = df[df['name'] == selected_restaurant].index[0]

# Display selected restaurant details
restaurant_info = df.iloc[selected_index]
st.write(f"**Name:** {restaurant_info['name']}")
st.write(f"**Restaurant Type:** {restaurant_info['rest_type']}")
st.write(f"**Cuisines Type:** {restaurant_info['cuisines']}")
st.write(f"**URL:** [{restaurant_info['name']}]({restaurant_info['url']})")

# Find similar restaurants using KNN
distances, indices = knn.kneighbors(X_scaled[selected_index].reshape(1, -1))

# Display recommendations in a table format
if indices.size > 0:
    st.subheader('Recommended Restaurants')
    recommended_restaurants = df.iloc[indices.flatten()]
    recommended_restaurants = recommended_restaurants[recommended_restaurants['name'] != selected_restaurant]
    
    if not recommended_restaurants.empty:
        # Display the dataframe using st.table
        st.table(recommended_restaurants[['name', 'rest_type', 'cuisines', 'url']].apply(lambda x: f"[{x['name']}]({x['url']})", axis=1))
    else:
        st.write("No recommendations available.")
else:
    st.write("No recommendations available.")
