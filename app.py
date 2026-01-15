import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
anime_df = pd.read_csv("anime.csv")

# Fill missing genres
anime_df['genre'] = anime_df['genre'].fillna('')

# Create TF-IDF vectorizer and compute cosine similarity
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(anime_df['genre'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create indices mapping
indices = pd.Series(anime_df.index, index=anime_df['name'])

# Define the recommendation function
def content_based_recommendation(title, top_n=5, sig=cosine_sim):
    if title not in anime_df['name'].values:
        return f"Anime '{title}' not found"

    # Get the index corresponding to original title
    index = indices[title]

    # Get pair-wise similarity score
    sim_scores = list(enumerate(sig[index]))

    # Sort the movies
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    # Movies indices
    anime_indices = [i[0] for i in sim_scores]

    return anime_df['name'].iloc[anime_indices]

# UI
st.title("Anime Recommendation System")
st.write("Enter an anime name to get recommendations based on content similarity.")

anime_name = st.text_input("Anime Name:")
top_n = st.slider("Number of recommendations:", 1, 20, 10)

if st.button("Get Recommendations"):
    if anime_name:
        recommendations = content_based_recommendation(anime_name, top_n)
        if isinstance(recommendations, str):
            st.error(recommendations)
        else:
            st.success(f"Recommendations for '{anime_name}':")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    else:
        st.warning("Please enter an anime name.")

