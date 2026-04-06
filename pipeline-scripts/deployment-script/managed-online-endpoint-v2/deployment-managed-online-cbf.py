import json
import os
import tempfile

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    CodeConfiguration,
    Environment,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
    ProbeSettings,
)
from azure.identity import DefaultAzureCredential

from azureml.core import Dataset, Workspace
from transformers import AutoTokenizer, AutoModel


ENDPOINT_NAME         = "cbf-movie-recommender"
DEPLOYMENT_NAME       = "cbf-deployment-v1"
MODEL_NAME            = "cbf-movie-recommender-model"

ARTIFACT_DIR          = "./model_artifacts"
ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


ws = Workspace.from_config()

ml_client = MLClient.from_config(credential=DefaultAzureCredential())

os.makedirs(ARTIFACT_DIR, exist_ok=True)


emb_ds = Dataset.get_by_name(ws, "movie_embeddings")
emb_ds.download(target_path=ARTIFACT_DIR, overwrite=True)


processed_ds = Dataset.get_by_name(ws, "cbf_processed_movies")
processed_df  = processed_ds.to_pandas_dataframe()
processed_csv_path = os.path.join(ARTIFACT_DIR, "processed_movies.csv")
processed_df.to_csv(processed_csv_path, index=False)

ratings_ds = Dataset.get_by_name(ws, "ratings_small")
ratings_ds.download(target_path=ARTIFACT_DIR, overwrite=True)
os.rename(
    os.path.join(ARTIFACT_DIR, "ratings_small.csv"),
    os.path.join(ARTIFACT_DIR, "ratings.csv"),
)

encoder_save_path = os.path.join(ARTIFACT_DIR, "minilm")

print("Downloading MiniLM model weights locally ...")
AutoTokenizer.from_pretrained(ENCODER_MODEL).save_pretrained(encoder_save_path)
AutoModel.from_pretrained(ENCODER_MODEL).save_pretrained(encoder_save_path)
print("  Done.")

model = Model(
    path=ARTIFACT_DIR,
    name=MODEL_NAME,
    description=(
        "CBF movie recommender artifacts: "
        "MiniLM embeddings (.npy) + metadata CSVs"
    ),
    type="custom_model",
)
registered_model = ml_client.models.create_or_update(model)

endpoint = ManagedOnlineEndpoint(
    name=ENDPOINT_NAME,
    description="Real-time CBF movie recommendations",
    auth_mode="key",
    tags={"project": "netflix-cbf"},
)
endpoint = ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print(f"  Provisioning state: {endpoint.provisioning_state}")


print(f"\nCreating deployment '{DEPLOYMENT_NAME}' ...")
inference_env = Environment(
    name="cbf-inference-env",
    description="Runtime for CBF scoring container",
    conda_file="conda-dependencies.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest",
)

deployment = ManagedOnlineDeployment(
    name=DEPLOYMENT_NAME,
    endpoint_name=ENDPOINT_NAME,
    model=registered_model.id,
    environment=inference_env,
    code_configuration=CodeConfiguration(
        code=".",
        scoring_script="score_managed_online.py",
    ),
    instance_type="Standard_F2s_v2",
    instance_count=1,
    liveness_probe=ProbeSettings(
        initial_delay=600,
        period=30,
        timeout=5,
        failure_threshold=3,
        success_threshold=1,
    ),
    readiness_probe=ProbeSettings(
        initial_delay=600,
        period=30,
        timeout=5,
        failure_threshold=3,
        success_threshold=1,
    ),
)

poller = ml_client.online_deployments.begin_create_or_update(deployment)
print("  Waiting for deployment to finish (10-15 min on first run) ...")
result = poller.result()
print(f"  Provisioning state: {result.provisioning_state}")


print("\nRouting 100% traffic to cbf-deployment-v1 deployment ...")
endpoint = ml_client.online_endpoints.get(ENDPOINT_NAME)
endpoint.traffic = {DEPLOYMENT_NAME: 100}
ml_client.online_endpoints.begin_create_or_update(endpoint).result()
print("  Done.")


def parse_response(raw: str) -> dict:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"error": "Something went wrong, please try again."}


print("\nRunning smoke tests ...")
 
test_payloads = [
    {"movie_title": "The Dark Knight", "top_k": 3},
    {"movie_title": "The Dark Knight", "top_k": 3, "user_id": 1},
    {"movie_index": 0,                 "top_k": 3, "user_id": 1},
    {"query": "space exploration and astronauts", "top_k": 3, "user_id": 1},
]

with tempfile.TemporaryDirectory() as tmp:
    for payload in test_payloads:
        payload_path = os.path.join(tmp, "payload.json")
        with open(payload_path, "w") as f:
            json.dump(payload, f)

        raw      = ml_client.online_endpoints.invoke(
            endpoint_name=ENDPOINT_NAME,
            deployment_name=DEPLOYMENT_NAME,
            request_file=payload_path,
        )
        response = parse_response(raw)

        print(f"\n  Input  : {payload}")
        if "error" in response:
            print(f"  ERROR  : {response['error']}")
        else:
            print(f"  Source : {response.get('source') or response.get('query')}")
            for rec in response.get("recommendations", []):
                print(f"    [{rec['rank']}] {rec['title']}  (sim={rec['similarity']})")

score_uri = ml_client.online_endpoints.get(ENDPOINT_NAME).scoring_uri

print("Endpoint deployed successfully.")
print(f"  Scoring URI : {score_uri}")