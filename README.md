# 🎬 CineMatch AI - Hybrid Movie Recommender

A movie recommendation system that combines **collaborative filtering** (user behavior)
and **content-based filtering** (genres), served with **FastAPI** and a colorful
aurora-glass UI.
## Screenshots

<p align="center">
  <img src="screen1.png" width="30%" alt="Toy Story recommendations">
  <img src="screen2.png" width="30%" alt="Matrix recommendations">
  <img src="screen3.png" width="30%" alt="Titanic recommendations">
</p>
<p align="center">
  <img src="screen4.png" width="30%" alt="Dark Knight recommendations">
  <img src="screen5.png" width="30%" alt="Inception recommendations">
</p>


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
