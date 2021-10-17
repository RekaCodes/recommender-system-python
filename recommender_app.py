import streamlit as st
st.set_page_config(page_title="Get Rec'd", layout="wide", initial_sidebar_state='collapsed')

import pandas as pd 
import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def main():

    # load movies data
    movies_meta = pd.read_csv('data/movies_metadata.csv', low_memory=False)

    # change numeric columns from object to float
    int_cols = ['budget', 'popularity', 'revenue', 'runtime', 'vote_average', 'vote_count']
    movies_meta[int_cols] = movies_meta[int_cols].apply(pd.to_numeric, errors='coerce')

    # replace NaNs with blanks ('')
    movies_meta['overview'] = movies_meta['overview'].fillna('')

    # defining vectorizer object
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_meta['overview'])

    # compute cosine similarity scores
    cos_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_recommendations(title, cos_sim = cos_sim):

        # get index of movie give a title
        idx = movies_meta[movies_meta['title']==title].sort_values(by='popularity', ascending=False)[:1].index.values[0]

        # get similarity scores for given movie
        sim_scores = list(enumerate(cos_sim[idx]))

        # sort based on score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # get scores of 10 similar movies, skipping index [0]
        sim_scores = sim_scores[1:11]

        # get movie indices
        movie_indices = [i[0] for i in sim_scores]

        # return top 10 most similar movies
        return movies_meta['title'].iloc[movie_indices]


    st.write("""
    # Get Rec'd
    A Recommendation System built in `python` + `streamlit` by [Derrick Green](https://derrickhudsongreen.wordpress.com)

    ---
    """)

    input_title = st.text_input(label='Insert Title Here.')

    if input_title:
        get_recommendations(input_title)




if __name__ == "__main__":
    main()