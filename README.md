# 🎬 CineMatch AI - Hybrid Movie Recommender

A movie recommendation system that combines **collaborative filtering** (user behavior)
and **content-based filtering** (genres), served with **FastAPI** and a colorful
aurora-glass UI.

## Features
- Hybrid, Collaborative and Content-based modes
- Live search with autocomplete
- Match-score bar on every recommendation
- Mean-centered item-item cosine similarity (handles strict vs generous raters)

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, FastAPI, HTML/CSS/JS

## How it works
1. Genres are converted with TF-IDF, then cosine similarity gives content similarity.
2. Ratings are mean-centered per user, then item-item cosine similarity gives collaborative similarity.
3. Hybrid score = 0.7 x collaborative + 0.3 x content.

## Dataset
MovieLens Latest Small (100k ratings, 600 users):
https://grouplens.org/datasets/movielens/latest/

## Run locally
    pip install -r requirements.txt
    uvicorn main:app --reload

Then open http://127.0.0.1:8000

## Author
Rafaqat - Karak, Pakistan
