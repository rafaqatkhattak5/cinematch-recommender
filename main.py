from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).parent

popular = joblib.load(BASE / "movies_df.pkl")
cf_sim = joblib.load(BASE / "cf_sim.pkl")
content_sim = joblib.load(BASE / "content_sim.pkl")
SIMS = {
    "hybrid": 0.7 * cf_sim + 0.3 * content_sim,
    "cf": cf_sim,
    "content": content_sim,
}
HTML = (BASE / "index.html").read_text(encoding="utf-8")

app = FastAPI(title="CineMatch AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML

@app.get("/search")
def search(q: str = Query(..., min_length=2)):
    m = (popular[popular["title"].str.contains(q, case=False, regex=False)]
         .sort_values("num_ratings", ascending=False).head(8))
    return [{"id": int(r.movieId), "title": r.title} for r in m.itertuples()]

@app.get("/recommend")
def recommend(movie_id: int, mode: str = "hybrid", n: int = 10):
    if mode not in SIMS:
        raise HTTPException(400, "Mode must be hybrid, cf or content")
    pos = popular.index[popular["movieId"] == movie_id]
    if len(pos) == 0:
        raise HTTPException(404, "Movie not found")
    idx = int(pos[0])
    sims = SIMS[mode][idx]
    top = [i for i in np.argsort(sims)[::-1] if i != idx][:n]
    results = []
    for i in top:
        row = popular.iloc[i]
        results.append({
            "title": row["title"],
            "genres": row["genres"],
            "year": str(row["year"]) if str(row["year"]) != "nan" else "",
            "avg_rating": float(row["avg_rating"]),
            "num_ratings": int(row["num_ratings"]),
            "score": float(sims[i]),
        })
    return {"movie": popular.iloc[idx]["title"], "mode": mode, "results": results}
