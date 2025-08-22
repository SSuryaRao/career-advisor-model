import vertexai
from vertexai.generative_models import GenerativeModel

def query_career_advisor(project_id: str, location: str, endpoint_id: str, prompt: str):
    """
    Sends a prompt to a fine-tuned Gemini model deployed on a Vertex AI Endpoint.

    Args:
        project_id: Your Google Cloud project ID.
        location: The region where your model is deployed.
        endpoint_id: The numeric ID of the endpoint where the model is deployed.
        prompt: The text prompt to send to the model.
    """
    try:
        # Initialize the Vertex AI SDK
        print(f"Initializing Vertex AI for project '{project_id}' in '{location}'...")
        vertexai.init(project=project_id, location=location)

        # Load the generative model from the endpoint.
        # This is the crucial step for a tuned Gemini model.
        print(f"Loading model from endpoint '{endpoint_id}'...")
        model = GenerativeModel(
    model_name=f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
)
        # Send the prompt to the model to generate content.
        print(f"\nSending prompt: '{prompt}'")
        print("Waiting for response...")
        response = model.generate_content(prompt)

        # Print the model's response text.
        if response.candidates:
            response_text = response.candidates[0].content.parts[0].text
            print("\n--- Model Response ---")
            print(response_text)
            print("----------------------\n")
        else:
            print("\nModel did not return any content.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nPlease ensure the following:")
        print("1. You have authenticated with 'gcloud auth application-default login'.")
        print("2. The project ID, location, and endpoint ID are correct.")
        print("3. The Vertex AI API is enabled in your Google Cloud project.")


if __name__ == '__main__':
    PROJECT_ID = "career-advisor-469806"
    LOCATION = "us-central1"
    # This is the correct ENDPOINT_ID from your gcloud list command
    ENDPOINT_ID = "5411957860122755072"

    example_prompt = "What are the most important skills for a software engineer in 2025?"

    query_career_advisor(PROJECT_ID, LOCATION, ENDPOINT_ID, example_prompt)