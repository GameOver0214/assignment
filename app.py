import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Streamlit app title
st.title('Restaurant Recommendation System')

# Load the dataset
try:
    df = pd.read_csv('zomato_extracted.csv')
    
    # Ensure 'cuisines' column is clean
    df['cuisines'] = df['cuisines'].fillna('')  # Fill NaN with empty string
    df['cuisines'] = df['cuisines'].astype(str)  # Ensure all entries are strings

except FileNotFoundError:
    st.error("Dataset file not found. Please ensure the 'zomato_extracted.csv' file is in the correct directory.")
    st.stop()  # Stop the execution of the script

except pd.errors.EmptyDataError:
    st.error("The dataset is empty. Please check the file content.")
    st.stop()  # Stop the execution of the script

except Exception as e:
    st.error(f"An error occurred while loading the dataset: {e}")
    st.stop()  # Stop the execution of the script

# Remove duplicate restaurant names for the dropdown
unique_restaurant_names = df['name'].drop_duplicates().tolist()

# One-hot encode the 'cuisines' column
df_encoded = pd.get_dummies(df, columns=['cuisines'], prefix='cuisine')

# Drop non-numeric columns (except for 'name', 'rest_type', and 'url')
non_numeric_columns = df_encoded.select_dtypes(exclude=['number']).columns
df_encoded.drop(columns=non_numeric_columns, inplace=True, errors='ignore')

# Check for NaN values and drop rows with NaN values
if df_encoded.isnull().values.any():
    st.warning("NaN values found in the features DataFrame. Dropping rows with NaN values.")
    df_encoded.dropna(inplace=True)

# Check if there are enough rows left for processing
if df_encoded.shape[0] < 1:
    st.error("No valid data available for processing. Please check the dataset.")
    st.stop()  # Stop the execution of the script

# Prepare features for scaling
X = df_encoded.drop(columns=['name', 'rest_type', 'url'], errors='ignore')

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit KNN model
knn = NearestNeighbors(n_neighbors=5, algorithm='auto')
knn.fit(X_scaled)

# Restaurant selection
st.subheader('Choose a Restaurant')
selected_restaurant = st.selectbox('Select a restaurant', unique_restaurant_names)

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
# Display recommendations in a table format
if indices.size > 0:
    st.subheader('Recommended Restaurants')
    recommended_restaurants = df.iloc[indices.flatten()]
    recommended_restaurants = recommended_restaurants[recommended_restaurants['name'] != selected_restaurant]
    
    if not recommended_restaurants.empty:
        # Create a new DataFrame to store formatted results
        formatted_recommendations = pd.DataFrame({
            'Name': recommended_restaurants['name'],
            'Rest Type': recommended_restaurants['rest_type'],
            'Cuisines': recommended_restaurants['cuisines'],
            'URL': recommended_restaurants['url'].apply(lambda x: f"[{x.split('/')[-1]}]({x})")  # Use last part of URL as the display name
        })
        
        # Display the formatted DataFrame using st.table
        st.table(formatted_recommendations)
    else:
        st.write("No recommendations available.")
else:
    st.write("No recommendations available.")

