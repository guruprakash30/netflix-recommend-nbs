import json
import tempfile
import os
 
from azureml.core import Workspace, Environment, Dataset    
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice
# Load workspace
ws = Workspace.from_config()

print("Downloading cbf_processed_movies dataset ...")
processed_ds = Dataset.get_by_name(ws, "cbf_processed_movies")
processed_df = processed_ds.to_pandas_dataframe()
 
lookup_csv_path = "./processed_movies_lookup.csv"
processed_df[["movieid", "title", "genres", "overview"]].to_csv(lookup_csv_path, index=False)

print("Registering processed_movies as lookup model ...")
lookup_model = Model.register(
    workspace=ws,
    model_name="cbf-processed-movies-lookup",
    model_path=lookup_csv_path,
    description="Movie metadata CSV for CF recommendation enrichment",
)

ratings_ds = Dataset.get_by_name(ws, "ratings_small")
ratings_ds.download(target_path=".", overwrite=True)
os.rename("ratings_small.csv", "./ratings_lookup.csv")

ratings_lookup_model = Model.register(
    workspace=ws,
    model_name="ratings-small-lookup",
    model_path="./ratings_lookup.csv",
    description="Full user ratings for seen-movie filtering",
)

# Get registered model (specific version)
cf_model  = Model(ws, name="netflix-recommender-cf-model",    version=2)
trainset  = Model(ws, name="netflix-recommender-cf-trainset", version=1)

# Create environment from conda YAML
deploy_env = Environment.from_conda_specification(
    name="netflix-cf-online-env",
    file_path="conda-environment.yaml"
)   

# Inference configuration
inference_config = InferenceConfig(
    entry_script="score.py",
    environment=deploy_env
)

# ACI deployment config
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=3,
    tags={"type": "online"},
    description="Netflix Collaborative filtering Online Endpoint"
)

# Service name
service_name = "netflix-cf-online"

# Deploy
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[cf_model, trainset, lookup_model, ratings_lookup_model],
    inference_config=inference_config,
    deployment_config=aci_config,
    overwrite=True   # allows redeploy without delete
)

service.wait_for_deployment(show_output=True)

# Now check logs
print(service.get_logs())

# ── Smoke tests ───────────────────────────────────────────────────────────────
def parse_response(raw: str) -> dict:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"error": "Something went wrong, please try again."}
 
 
print("\nRunning smoke tests ...")
 
test_payloads = [
    {"userId": 1,  "top_k": 3},
    {"userId": 42, "top_k": 3},
]
 
for payload in test_payloads:
    raw      = service.run(input_data=json.dumps(payload))
    response = parse_response(raw) if isinstance(raw, str) else raw
 
    print(f"\n  Input  : {payload}")
    if "error" in response:
        print(f"  ERROR  : {response['error']}")
    else:
        print(f"  User   : {response.get('userId')}")
        for rec in response.get("recommendations", []):
            print(
                f"    [{rec['rank']}] {rec['title']}"
                f"  |  {rec['genres']}"
                f"  |  Love: {rec['you_will_love']}"
            )
 
print("\nEndpoint deployed successfully.")
print(f"  Scoring URI : {service.scoring_uri}")