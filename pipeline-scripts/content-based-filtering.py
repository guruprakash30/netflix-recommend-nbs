    # ## Import Libraries
import pandas as pd
import ast
import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import random
from azureml.core import Run
from azureml.data.dataset_factory import FileDatasetFactory

import numpy as np
from azureml.core import Dataset, Workspace, Datastore

# Start an Azure ML run
run = Run.get_context()
ws = run.experiment.workspace

# Access datasets Mounted paths via input_datasets dictionary
movies_path = run.input_datasets['movies']
keywords_path = run.input_datasets['keywords']


# Load CSVs
movies = pd.read_csv(movies_path, low_memory=False)
keywords = pd.read_csv(keywords_path, low_memory=False)

# Select relevant columns and rename id -> movieid
movies = movies[['id', 'genres', 'overview', 'title']].rename(columns={'id': 'movieid'})
keywords = keywords[['id', 'keywords']].rename(columns={'id': 'movieid'})

# ## Parse JSON Columns
def parse_genres(text):
    try:
        genres = ast.literal_eval(text)
        return ", ".join([g['name'] for g in genres])
    except:
        return ""

movies['genres'] = movies['genres'].apply(parse_genres)

def parse_keywords(text):
    try:
        kws = ast.literal_eval(text)
        return ", ".join([k['name'] for k in kws])
    except:
        return ""

keywords['keywords'] = keywords['keywords'].apply(parse_keywords)

# cleanup data
movies['movieid'] = pd.to_numeric(movies['movieid'], errors='coerce').astype('Int64')
keywords['movieid'] = pd.to_numeric(keywords['movieid'], errors='coerce').astype('Int64')

movies = movies.dropna(subset=['movieid']).reset_index(drop=True)
keywords = keywords.dropna(subset=['movieid']).reset_index(drop=True)

# ## Merge Movies & Keywords
df = movies.merge(keywords, on='movieid', how='inner')
# delete all empty string columns as well
df[['overview', 'genres', 'keywords', 'title']] = df[['overview', 'genres', 'keywords', 'title']].replace('', np.nan)

# Drop rows with missing overview
df = df.dropna(subset=['overview', 'genres', 'keywords', 'title'], how='any').reset_index(drop=True)

# ## Create Structured "Soup"
def create_soup(row):
    return (
        f"Overview: {row['overview']}. "
        f"Genres: {row['genres']}. "
        f"Keywords: {row['keywords']}."
    )

df['soup'] = df.apply(create_soup, axis=1)


# ## Load Pretrained Sentence Transformer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Embedding function
def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)
    embedding = torch.nn.functional.normalize(embedding, dim=1)
    return embedding


# ## Build Embedding Matrix
embeddings = torch.vstack([embed(text) for text in df['soup']])
print("Embedding matrix shape:", embeddings.shape)  # (num_movies, 384)

# ## Recommend Function
def recommend(movie_idx, top_k=2):
    """
    Recommend top_k similar movies for a given movie index.
    """
    target = embeddings[movie_idx]
    sims = cosine_similarity(target.unsqueeze(0), embeddings)
    top_indices = torch.topk(sims, top_k + 1).indices[1:]
    return df.iloc[top_indices]

# ## Genre Overlap Metric
def genre_overlap(movie_idx, rec_indices, log_prefix="genre_overlap"):
    target_genres = set(df.iloc[movie_idx]['genres'].split(", "))
    scores = []
    for idx in rec_indices:
        rec_genres = set(df.iloc[idx]['genres'].split(", "))
        overlap = len(target_genres & rec_genres) / len(target_genres) if target_genres else 0
        scores.append(overlap)
        print(f"Target Genres: {target_genres}, Recommended Genres: {rec_genres}, Overlap: {overlap:.2f}")

    avg_overlap = sum(scores) / len(scores) if scores else 0
    # Log with unique name to prevent overwrite
    run.log(f"{log_prefix}_{random.randint(0,9999)}", avg_overlap)
    return avg_overlap


# ## Intra-Similarity Metric
def intra_similarity(indices, log_prefix="intra_sim"):
    sims = []
    for i in indices:
        for j in indices:
            if i < j:  # avoid repeating symmetric comparisons
                sims.append(cosine_similarity(
                    embeddings[i].unsqueeze(0),
                    embeddings[j].unsqueeze(0)
                ).item())
    avg_sim = sum(sims) / len(sims) if sims else 0
    # Log with unique name
    run.log(f"{log_prefix}_{random.randint(0,9999)}", avg_sim)
    return avg_sim

# Pick 3 random movie indices
movie_indices = random.sample(range(len(df)), 3)

for i, movie_idx in enumerate(movie_indices):
    print(f"\n=== Movie {i+1} (Index: {movie_idx}) ===")
    print("Movie Title:", df.iloc[movie_idx]['title'])

    # Get recommendations
    recommendations = recommend(movie_idx, top_k=5)
    print("Recommended Movies:")
    print(recommendations[['title', 'genres', 'keywords', 'overview']])

    rec_indices = recommendations.index.tolist()

    # Genre Overlap
    print("Genre Overlap:", genre_overlap(movie_idx, rec_indices, log_prefix=f"genre_overlap_{movie_idx}"))

    # Embedding Cohesion
    print("Embedding Cohesion:", intra_similarity(rec_indices, log_prefix=f"intra_sim_{movie_idx}"))


# ## Save and Upload Embeddings

# Convert tensor to numpy array
emb_array = embeddings.cpu().numpy()  # shape (num_movies, 384)


# Save locally
np.save("movie_embeddings.npy", embeddings.cpu().numpy())

datastore = ws.get_default_datastore()

datastore.upload_files(
    files=["./movie_embeddings.npy"],
    target_path="embeddings/",
    overwrite=True,
    show_progress=True
)


emb_dataset = Dataset.File.from_files(
    path=(datastore, "embeddings/movie_embeddings.npy")
)
emb_dataset.register(
    workspace=ws,
    name="movie_embeddings",
    description="CBF embeddings for all movies",
    create_new_version=True
)


df[['movieid', 'title', 'genres', 'keywords', 'overview', 'soup']].to_csv("processed_movies.csv", index=False)

datastore.upload_files(
    files=["./processed_movies.csv"],
    target_path="processed_movies/",
    overwrite=True,
    show_progress=True
)

Dataset.Tabular.from_delimited_files(
    path=(datastore, "processed_movies/processed_movies.csv")
).register(workspace=ws, name="cbf_processed_movies", create_new_version=True)

run.complete()