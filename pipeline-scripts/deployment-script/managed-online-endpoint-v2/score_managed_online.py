import os
import json
import numpy as np
import torch
import pandas as pd
from torch.nn.functional import cosine_similarity
from transformers import AutoModel, AutoTokenizer


# ── Globals ──────────────────────────────────────────────
embeddings = None   # torch.Tensor  (num_movies, 384)
df         = None   # DataFrame with title / genres / keywords / overview
tokenizer  = None   # HuggingFace tokenizer
encoder    = None   # HuggingFace model for on-the-fly query encoding
user_seen_idx = {}  # dict[int, set[int]]  userId → set of df row indices

def encode_query(text: str) -> torch.Tensor:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = encoder(**inputs)
    pooled = output.last_hidden_state.mean(dim=1) 
    normed = torch.nn.functional.normalize(pooled, dim=1)
    return normed.squeeze(0)


def top_k_recs(target: torch.Tensor, top_k: int, exclude_idx: int | None = None, seen_idx: set | None = None) -> list[dict]:
    sims = cosine_similarity(target.unsqueeze(0), embeddings)   # (num_movies,)

    if exclude_idx is not None:
        sims[exclude_idx] = -2.0

    if seen_idx:
        for idx in seen_idx:
            sims[idx] = -2.0    

    top_indices = torch.topk(sims, top_k).indices.tolist()

    recommendations = []
    for rank, idx in enumerate(top_indices, start=1):
        row = df.iloc[idx]
        recommendations.append({
            "rank":       rank,
            "title":      row["title"],
            "genres":     row["genres"],
            "overview":   row["overview"],
            "similarity": round(sims[idx].item(), 4),
        })
    return recommendations


def init():
    global embeddings, df, tokenizer, encoder

    model_dir = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model_artifacts")
    encoder_path  = os.path.join(model_dir, "minilm")

    # ── Load pre-computed embeddings ──────────────────────────────────────
    emb_path   = os.path.join(model_dir, "movie_embeddings.npy")
    emb_np     = np.load(emb_path)                              # (num_movies, 384)
    embeddings = torch.tensor(emb_np, dtype=torch.float32)

    df = pd.read_csv(os.path.join(model_dir, "processed_movies.csv"))

     # Build userId → set-of-df-row-indices from ratings.csv
    ratings = pd.read_csv(
        os.path.join(model_dir, "ratings.csv"),
        usecols=["userId", "movieId"],
    )
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(df["movieid"])}
    for user_id, grp in ratings.groupby("userId"):
        user_seen_idx[int(user_id)] = {
        movie_id_to_idx[mid]
        for mid in grp["movieId"]
        if mid in movie_id_to_idx
        }

    tokenizer = AutoTokenizer.from_pretrained(encoder_path)
    encoder   = AutoModel.from_pretrained(encoder_path)
    encoder.eval()


def run(raw_data: str) -> str:
    """
    Accepted request bodies (JSON):
 
      { "movie_title":  "The Dark Knight",              "top_k": 5, "user_id": 42 }
      { "movie_index":  42,                             "top_k": 5, "user_id": 42 }
      { "query":        "space exploration astronauts", "top_k": 5, "user_id": 42 }
 
    'user_id' is optional. When supplied, movies the user already rated
    are excluded from the returned top_k.
    """
    try:
        payload  = json.loads(raw_data)
        top_k   = max(1, int(payload.get("top_k", 5)))
        user_id  = payload.get("user_id")
        seen_idx = user_seen_idx.get(int(user_id), set()) if user_id is not None else set()


        # ── Resolve target embedding ──────────────────────────────────────
        if "movie_title" in payload:
            title   = payload["movie_title"].strip().lower()
            matches = df[df["title"].str.lower().str.contains(title, na=False)]
            if matches.empty:
                return {"error": f"No movie found matching '{payload['movie_title']}'"}
            movie_idx    = int(matches.index[0])
            target       = embeddings[movie_idx]
            source_label = df.iloc[movie_idx]["title"]

            return {
                "source":          source_label,
                "recommendations": top_k_recs(target, top_k, exclude_idx=movie_idx, seen_idx = seen_idx),
            }

        elif "movie_index" in payload:
            movie_idx = int(payload["movie_index"])
            if not (0 <= movie_idx < len(df)):
                return {"error": f"movie_index {movie_idx} out of range (0–{len(df)-1})"}
            target       = embeddings[movie_idx]
            source_label = df.iloc[movie_idx]["title"]

            return {
                "source":          source_label,
                "recommendations": top_k_recs(target, top_k, exclude_idx=movie_idx, seen_idx = seen_idx),
            }

        elif "query" in payload:
            query_text = payload["query"].strip()
            if not query_text:
                return {"error": "'query' must be a non-empty string."}
 
            target = encode_query(query_text)
 
            return {
                "query":           query_text,
                "recommendations": top_k_recs(target, top_k, seen_idx = seen_idx),
            }

        else:
            return {
                "error": (
                    "Provide one of 'movie_title', 'movie_index', "
                    "or 'query' in the request body."
                )
            }

    except Exception as exc:
        return {"error": str(exc)}