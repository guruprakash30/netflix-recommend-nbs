import json
import joblib
import os
import pandas as pd
from azureml.core.model import Model

model    = None
trainset = None
movies_df = None
ratings_seen = {}

def init():
    global model, trainset, movies_df,ratings_seen
    
    try:
        model_path    = Model.get_model_path("netflix-recommender-cf-model", version=2)
        trainset_path = Model.get_model_path("netflix-recommender-cf-trainset", version=1)
        movies_path   = Model.get_model_path("cbf-processed-movies-lookup")
 
        model    = joblib.load(model_path)
        trainset = joblib.load(trainset_path)
 
        movies_df = pd.read_csv(
            movies_path,
            usecols=["movieid", "title", "genres", "overview"]
        ).set_index("movieid")   # index by movieid for O(1) lookup
 
        print("Model, trainset and movies lookup loaded successfully.")

        ratings_path = Model.get_model_path("ratings-small-lookup")
        ratings_raw  = pd.read_csv(ratings_path, usecols=["userId", "movieId"])
        for user_id, grp in ratings_raw.groupby("userId"):
            ratings_seen[int(user_id)] = set(grp["movieId"])
 
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
 

def get_top_k(user_id, top_k):
    seen_raw_ids = ratings_seen.get(int(user_id), set())
    
    all_movie_raw_ids = [
        trainset.to_raw_iid(iid) for iid in trainset.all_items()
    ]
 
    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in all_movie_raw_ids
        if movie_id not in seen_raw_ids
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
 
    results = []
    for rank, (movie_id, rating) in enumerate(predictions[:top_k], start=1):
        # SVD ratings are on a 0–5 scale → convert to a 0–100% love score
        love_pct = round((rating / 5) * 100, 1)
 
        meta = movies_df.loc[movie_id] if movie_id in movies_df.index else None
 
        results.append({
            "rank":          rank,
            "title":         meta["title"]    if meta is not None else f"Movie {movie_id}",
            "genres":        meta["genres"]   if meta is not None else "N/A",
            "overview":      meta["overview"] if meta is not None else "N/A",
            "you_will_love": f"{love_pct}%",
        })
 
    return results



def run(raw_data):
    try:
        data    = json.loads(raw_data)
        user_id = data.get("userId")
        top_k   = max(1, int(data.get("top_k", 10)))
 
        if user_id is None:
            return {"error": "userId is required."}
 
        recommendations = get_top_k(user_id, top_k)
 
        return {
            "userId":          user_id,
            "recommendations": recommendations,
        }
 
    except Exception as e:
        return {"error": str(e)}