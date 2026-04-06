# score_batch.py
import pandas as pd
import joblib
import argparse
import os
from azureml.core import Run, Model

def get_top_k(model, trainset, user_id, k=10):
    all_movie_inner_ids = trainset.all_items()
    all_movie_raw_ids = [trainset.to_raw_iid(iid) for iid in all_movie_inner_ids]

    predictions = [
        (movie_id, model.predict(user_id, movie_id).est)
        for movie_id in all_movie_raw_ids
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top_k = predictions[:k]
    return [{"movieId": mid, "predicted_rating": rating} for mid, rating in top_k]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_ids_file", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    # Resolve registered models on remote compute
    ws = Run.get_context().experiment.workspace
    model_path    = Model.get_model_path("netflix-recommender-cf-model", version=2, _workspace=ws)
    trainset_path = Model.get_model_path("netflix-recommender-cf-trainset", version=1, _workspace=ws)

    model    = joblib.load(model_path)
    trainset = joblib.load(trainset_path)

    # Load user IDs
    user_ids = pd.read_csv(args.user_ids_file)["userId"].tolist()

    all_recs = []
    for user_id in user_ids:
        recs = get_top_k(model, trainset, user_id, k=10)
        for r in recs:
            all_recs.append({"userId": user_id, **r})

    out_df = pd.DataFrame(all_recs)
    os.makedirs(args.output_path, exist_ok=True)
    out_file = os.path.join(args.output_path, "batch_recommendations.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Batch recommendations saved to {out_file}")

if __name__ == "__main__":
    main()