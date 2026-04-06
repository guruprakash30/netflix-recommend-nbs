from azureml.core import Workspace, ComputeTarget

ws = Workspace.from_config()

compute_name = "gururpakash28749-cc-01"

# Get existing compute cluster
compute_target = ComputeTarget(workspace=ws, name=compute_name)
print(f"Using compute cluster: {compute_target.name}")

from azureml.core import Environment


env = Environment.get(workspace=ws, name="netflix-recommend-cf-env-01", version="1")

from azureml.core.runconfig import RunConfiguration

run_config = RunConfiguration()
run_config.environment = env

from azureml.core import Dataset
from azureml.pipeline.steps import PythonScriptStep

dataset = Dataset.get_by_name(ws, "ratings_small")

train_step = PythonScriptStep(
    name="train-model",
    script_name="netflix-recommend-cf.py",
    arguments=[
        "--input-data", dataset.as_named_input("ratings").as_mount()
    ],
    compute_target=compute_target,
    source_directory=".",
    runconfig=run_config
)

from azureml.pipeline.core import Pipeline
from azureml.core import Experiment

pipeline = Pipeline(workspace=ws, steps=[train_step])

experiment = Experiment(ws, "netflix-recommender-exp-01")
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)