from google import genai

client = genai.Client(project="career-advisor-469806", location="us-central1", vertexai=True)

print("Supported base models for tuning:")

for model in client.models.list():
    if model.supported_actions and "createTunedModel" in model.supported_actions:
        print(model.name)
