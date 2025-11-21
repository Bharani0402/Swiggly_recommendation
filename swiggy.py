import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Load the datasets
# -------------------------

clean_df = pd.read_csv(r"C:\Users\Eshwar\OneDrive\Desktop\VS_Code\Swiggy\cleaned_data.csv")
enc_df = pd.read_csv(r"C:\Users\Eshwar\OneDrive\Desktop\VS_Code\Swiggy\encoded_data.csv")


st.title("Restaurant Recommendation App")

clean_df['city'] = clean_df['city'].astype(str)
clean_df['rating'] = pd.to_numeric(clean_df['rating'], errors='coerce')
clean_df['cost'] = pd.to_numeric(clean_df['cost'], errors='coerce')
# --- USER INPUT ---
st.sidebar.header("User Preferences")
city = st.sidebar.selectbox("Select City", sorted(clean_df['city'].dropna().unique()))
cuisine = st.sidebar.selectbox("Preferred Cuisine", sorted(clean_df['cuisine'].dropna().unique()))
min_rating = st.sidebar.slider("Minimum Rating", 0.0, 5.0, 0.0, step=0.5)
max_price = st.sidebar.number_input("Maximum Price (₹)", min_value=0, value=500, step=100)


# --- FILTER CLEANED DATA ---
filtered = clean_df[
(clean_df['city'] == city) &
(clean_df['rating'] >= min_rating) &
(clean_df['cost'] <= max_price)
]


if cuisine:
    filtered = filtered[filtered['cuisine'].str.contains(cuisine, case=False, na=False)]


    st.subheader("Matching Restaurants (Before Similarity Ranking)")
    st.dataframe(filtered.head(20))


# --- COSINE SIMILARITY ---
if not filtered.empty:
    encoded_filtered = enc_df.loc[filtered.index]
    reference_vector = encoded_filtered.iloc[[0]]
    similarity_scores = cosine_similarity(reference_vector, encoded_filtered).flatten()


    filtered['similarity'] = similarity_scores
    final = filtered.sort_values(by='similarity', ascending=False)


    st.subheader("Recommended Restaurants")
    st.dataframe(final[['name', 'city', 'cuisine', 'rating', 'cost', 'similarity']].head(10))
else:
    st.warning("No restaurants found for selected filters.")