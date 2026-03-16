import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

# -------------------------
# Load datasets with caching
# -------------------------
@st.cache_data
def load_data():
    clean_df = pd.read_csv(
        r"C:\Users\Eshwar\OneDrive\Desktop\VS_Code\Swiggy\cleaned_data.csv"
    )

    enc_df = sparse.load_npz(
        r"C:\Users\Eshwar\OneDrive\Desktop\VS_Code\Swiggy\encoded_data.npz"
    )

    return clean_df, enc_df


clean_df, enc_df = load_data()

# -------------------------
# Basic preprocessing
# -------------------------
clean_df['city'] = clean_df['city'].astype(str)
clean_df['rating'] = pd.to_numeric(clean_df['rating'], errors='coerce')
clean_df['cost'] = pd.to_numeric(clean_df['cost'], errors='coerce')


# -------------------------
# Streamlit UI
# -------------------------
st.title("🍽️ Restaurant Recommendation App")

st.sidebar.header("User Preferences")
city_options = ["All"] + sorted(clean_df['city'].dropna().unique())
city = st.sidebar.selectbox(
    "Select City",
    city_options
)

cuisine_options = ["All"] + sorted(clean_df['cuisine'].dropna().unique())
cuisine = st.sidebar.selectbox(
    "Preferred Cuisine",
    cuisine_options
)

min_rating = st.sidebar.slider(
    "Minimum Rating",
    0.0,
    5.0,
    0.0,
    step=0.5
)

max_price = st.sidebar.number_input(
    "Maximum Price (₹)",
    min_value=0,
    value=500,
    step=100
)

# -------------------------
# Filter dataset
# -------------------------
# st.write("Total rows:", len(clean_df))

city_rows = clean_df[clean_df['city'] == city]
# st.write("Rows after city filter:", len(city_rows))

rating_rows = city_rows[city_rows['rating'] >= min_rating]
# st.write("Rows after rating filter:", len(rating_rows))

price_rows = rating_rows[rating_rows['cost'] <= max_price]
# st.write("Rows after price filter:", len(price_rows))

if cuisine != "All":
    cuisine_rows = price_rows[
        price_rows['cuisine'].str.contains(cuisine, case=False, na=False)
    ]
   # st.write("Rows after cuisine filter:", len(cuisine_rows))
price_limit = max_price + 200
filtered = clean_df[
    (clean_df['city'] == city) &
    (clean_df['rating'] >= min_rating) &
    (clean_df['cost'] <= price_limit)
]

if cuisine != "All":
    filtered = filtered[
        filtered['cuisine'].str.contains(cuisine, case=False, na=False)
    ]
st.write("Filtered rows:", len(filtered))
st.subheader("Matching Restaurants (Before Ranking)")
st.dataframe(filtered.head(20))

# -------------------------
# Recommendation using cosine similarity
# -------------------------
if not filtered.empty:

    filtered = filtered.copy()

    # Get encoded rows using index
    encoded_filtered = enc_df[filtered.index]

    # First restaurant as reference
    reference_vector = encoded_filtered[0]

    similarity_scores = cosine_similarity(
        reference_vector,
        encoded_filtered
    ).flatten()

    filtered['similarity'] = similarity_scores

    final = filtered.sort_values(
        by='similarity',
        ascending=False
    )

    st.subheader("⭐ Recommended Restaurants")

    st.dataframe(
        final[
            ['name', 'city', 'cuisine', 'rating', 'cost', 'similarity']
        ].head(10)
    )

else:
    st.warning("⚠️ No restaurants found for selected filters.")
