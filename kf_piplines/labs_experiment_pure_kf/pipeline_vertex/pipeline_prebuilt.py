
"""Kubeflow Covertype Pipeline."""
import os


#https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.HyperparameterTuningJob
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from google_cloud_pipeline_components.aiplatform import (
    EndpointCreateOp,
    ModelDeployOp,
    ModelUploadOp,
)
from google_cloud_pipeline_components.experimental import (
    hyperparameter_tuning_job,
)

from kfp.v2 import dsl

PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

TRAINING_CONTAINER_IMAGE_URI = os.getenv("TRAINING_CONTAINER_IMAGE_URI")
SERVING_CONTAINER_IMAGE_URI = os.getenv("SERVING_CONTAINER_IMAGE_URI")
SERVING_MACHINE_TYPE = os.getenv("SERVING_MACHINE_TYPE", "n1-standard-16")

MAX_TRIAL_COUNT = int(os.getenv("MAX_TRIAL_COUNT", "5"))
PARALLEL_TRIAL_COUNT = int(os.getenv("PARALLEL_TRIAL_COUNT", "5"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))

PIPELINE_NAME = os.getenv("PIPELINE_NAME", "covertype")
BASE_OUTPUT_DIR = os.getenv("BASE_OUTPUT_DIR", PIPELINE_ROOT)
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", PIPELINE_NAME)

@dsl.pipeline(
    name=f"{PIPELINE_NAME}-kfp-pipeline",
    description="Kubeflow pipeline that tunes, trains, and deploys on Vertex",
    pipeline_root=PIPELINE_ROOT,
)
def create_pipeline():

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
                "accelerator_type": "NVIDIA_TESLA_T4",
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": TRAINING_CONTAINER_IMAGE_URI,
                "args": [
                    "--hptune",
                ],
            },
        }
    ]
    
    metric_spec = hyperparameter_tuning_job.serialize_metrics(
        {"accuracy": "maximize"}
    )

    parameter_spec = hyperparameter_tuning_job.serialize_parameters(
        {
            "alpha": hpt.DoubleParameterSpec(
                min=1.0e-4, max=1.0e-1, scale="linear"
            ),
            "max_iter": hpt.DiscreteParameterSpec(
                values=[1, 2], scale="linear"
            ),
        }
    )
    
    hp_tuning_task = hyperparameter_tuning_job.HyperparameterTuningJobRunOp(
        display_name=f"{PIPELINE_NAME}-kfp-tuning-job",
        project=PROJECT_ID,
        location=REGION,
        worker_pool_specs=worker_pool_specs,
        study_spec_metrics=metric_spec,
        study_spec_parameters=parameter_spec,
        max_trial_count=MAX_TRIAL_COUNT,
        parallel_trial_count=PARALLEL_TRIAL_COUNT,
        base_output_directory=PIPELINE_ROOT,
    )
    
    trials_task = hyperparameter_tuning_job.GetTrialsOp(
        gcp_resources=hp_tuning_task.outputs["gcp_resources"]
    )

    best_hyperparameters_task = (
        hyperparameter_tuning_job.GetBestHyperparametersOp(
            trials=trials_task.output, study_spec_metrics=metric_spec
        )
    )
    
    


    
    
