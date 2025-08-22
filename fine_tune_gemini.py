from google import genai
from google.genai import types
import time

PROJECT_ID = "career-advisor-469806"
LOCATION = "us-central1"
TRAINING_DATA_URI = "gs://ai-career-advisor-data-surya-123/career_advisor_dataset.jsonl"
BASE_MODEL = "gemini-2.5-flash"
TUNED_MODEL_DISPLAY_NAME = "gemini-career-advisor-v1"

# Initialize the GenAI client with GCP parameters
client = genai.Client(project=PROJECT_ID, location=LOCATION, vertexai=True)

print("GenAI Client initialized.")

training_dataset = types.TuningDataset(
    gcs_uri=TRAINING_DATA_URI
)

tuning_job = client.tunings.tune(
    base_model=BASE_MODEL,
    training_dataset=training_dataset,
    config=types.CreateTuningJobConfig(
        epoch_count=3,
        tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
    ),
)

print("Tuning job submitted:", tuning_job.name)

# Monitor until the tuning job completes
while tuning_job.state in {"JOB_STATE_PENDING", "JOB_STATE_RUNNING"}:
    print("Job state:", tuning_job.state)
    tuning_job = client.tunings.get(name=tuning_job.name)
    time.sleep(60)

print("Final status:", tuning_job.state)
if tuning_job.state == "JOB_STATE_SUCCEEDED":
    print("Fine-tuning completed successfully!")
    print("Tuned Model Endpoint:", tuning_job.tuned_model.endpoint)
else:
    print("Tuning job failed or was cancelled.")
