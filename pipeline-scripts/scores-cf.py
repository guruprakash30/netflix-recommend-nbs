import json
import joblib
import numpy as np

model = None
init_error = None  # store error if model loading fails

def init():
    global model, init_error
    model_path = "model.pkl"
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        init_error = str(e)
        print(f"Error loading model in init(): {init_error}")
        # Optionally raise exception to make container fail immediately
        raise e  # AzureML needs this to mark deployment failed

def run(raw_data):
    if init_error:
        # return error if model never loaded
        return {"error": f"Model not loaded: {init_error}"}
    
    try:
        data = json.loads(raw_data)
        user_id = data.get("userId")
        movie_id = data.get("movieId")
        if user_id is None or movie_id is None:
            return {"error": "userId and movieId required"}

        pred = model.predict(user_id, movie_id, r_ui=0)
        return {"userId": user_id, "movieId": movie_id, "predicted_rating": pred.est}
    except Exception as e:
        return {"error": str(e)}