import argparse
import os
import pandas as pd
from azureml.core import Run
import joblib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str)
args = parser.parse_args()

data_path = args.input_data
df = pd.read_csv(data_path)  # lazy loading from mount

ratings = df[['userId', 'movieId', 'rating']]

# Custom split: 80% per movie
train_list = []
test_list = []

for movie_id, group in ratings.groupby("movieId"):
    group = group.sample(frac=1, random_state=42)  # shuffle
    
    split_idx = int(np.ceil(0.8 * len(group)))
    
    train_list.append(group.iloc[:split_idx])
    test_list.append(group.iloc[split_idx:])

train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

print("Train size:", len(train_df))
print("Test size:", len(test_df))


# ## Convert to Surprise Format


from surprise import Dataset, Reader

reader = Reader(rating_scale=(1, 5))

train_data = Dataset.load_from_df(train_df, reader)
full_trainset = train_data.build_full_trainset()

# testset should be list of tuples
testset = list(test_df.itertuples(index=False, name=None))


# ## GridSearchCV on ONLY 80% Training Data


from surprise import SVD
from surprise.model_selection import GridSearchCV

param_grid = {
    "n_factors": [20, 50, 100, 150],
    "n_epochs": [10, 20],
    "lr_all": [0.002, 0.005],
    "reg_all": [0.02, 0.1]
}

gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=5, n_jobs=2)

# IMPORTANT: pass ONLY train_data (80%)
gs.fit(train_data)

run = Run.get_context()


run.log("Best RMSE:", gs.best_score["rmse"])
run.log("Best Params:", gs.best_params["rmse"])


# ## Train Final Model with Best Params


best_params = gs.best_params["rmse"]

model = SVD(
    n_factors=best_params["n_factors"],
    n_epochs=best_params["n_epochs"],
    lr_all=best_params["lr_all"],
    reg_all=best_params["reg_all"]
)

model.fit(full_trainset)


# ## Predict on 20% Test Set


predictions = model.test(testset)


# ## RMSE Evaluation



from surprise import accuracy

rmse = accuracy.rmse(predictions)
run.log("Final RMSE:", rmse)


# ## Precision, Recall, F1, Accuracy (Your Custom Logic)


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

y_true = []
y_pred = []

for pred in predictions:
    actual_rating = pred.r_ui
    estimated_rating = pred.est

    actual_label = 1 if actual_rating >= 4 else 0
    predicted_label = 1 if estimated_rating >= 4 else 0

    y_true.append(actual_label)
    y_pred.append(predicted_label)

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy_val = accuracy_score(y_true, y_pred)

run.log("Precision:", precision)
run.log("Recall:", recall)
run.log("F1 Score:", f1)
run.log("Accuracy:", accuracy_val)

os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/model.pkl")
joblib.dump(full_trainset, "outputs/trainset.pkl")

run.upload_file(name="outputs/model.pkl", path_or_stream="outputs/model.pkl")
run.upload_file(name="outputs/trainset.pkl", path_or_stream="outputs/trainset.pkl")

run.register_model(
    model_name="netflix-recommender-cf-model",
    model_path="outputs/model.pkl"
)

run.register_model(
    model_name="netflix-recommender-cf-trainset",
    model_path="outputs/trainset.pkl"
)

