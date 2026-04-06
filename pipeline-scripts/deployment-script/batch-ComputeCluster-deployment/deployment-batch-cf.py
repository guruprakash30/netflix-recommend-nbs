from azureml.core import Workspace, Experiment, Dataset, Environment
from azureml.core.compute import ComputeTarget
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.data import OutputFileDatasetConfig

# Workspace
ws = Workspace.from_config()

# Compute
compute_name = "gururpakash28749-cc-01"
compute_target = ComputeTarget(ws, name=compute_name)
print(f"Using compute cluster: {compute_target.name}")

# Environment
env = Environment.get(workspace=ws, name="netflix-recommend-cf-env-01", version="1")
run_config = RunConfiguration()
run_config.environment = env

# Datasets
user_ids_ds = Dataset.get_by_name(ws, "user_ids_only").as_download()  # passed as download

batch_output = OutputFileDatasetConfig(
    name="batch_output",
    destination=(ws.get_default_datastore(), "batch_outputs/")
).as_upload(overwrite=True)

batch_step = PythonScriptStep(
    name="batch-inference",
    script_name="score_batch.py",
    arguments=[
        "--user_ids_file", user_ids_ds,
        "--output_path",   batch_output,
    ],
    outputs=[batch_output],
    compute_target=compute_target,
    source_directory=".",
    runconfig=run_config,
    allow_reuse=False
)

# Pipeline
pipeline = Pipeline(workspace=ws, steps=[batch_step])
experiment = Experiment(ws, "netflix-recommender-batch")
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)