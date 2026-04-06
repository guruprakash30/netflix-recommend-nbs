# pipeline_submit.py
from azureml.core import Workspace, ComputeTarget, Dataset, Experiment, Environment
from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration

# Connect to workspace
ws = Workspace.from_config()

compute_name = "gururpakash28749-cc-01"
compute_target = ComputeTarget(workspace=ws, name=compute_name)
print(f"Using compute cluster: {compute_target.name}")

# Get environment
env = Environment.get(workspace=ws, name="netflix-recommend-cbf-env-02", version="10")

# Run configuration
run_config = RunConfiguration()
run_config.environment = env

# Get datasets
movies_ds = Dataset.get_by_name(ws, "movies_metadata")
keywords_ds = Dataset.get_by_name(ws, "keywords")

# Define PythonScriptStep using inputs
train_step = PythonScriptStep(
    name="train-cbf-model",
    script_name="content-based-filtering.py",
    inputs=[
        movies_ds.as_named_input("movies").as_mount(),
        keywords_ds.as_named_input("keywords").as_mount()
    ],
    compute_target=compute_target,
    source_directory=".",  # script location
    runconfig=run_config,
    allow_reuse=True
)

# Build pipeline
pipeline = Pipeline(workspace=ws, steps=[train_step])
experiment = Experiment(ws, "netflix-recommender-exp-01")
pipeline_run = experiment.submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)