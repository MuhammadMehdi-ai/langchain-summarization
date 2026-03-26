from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("AZURE_OPENAI_API_KEY"))
print(os.getenv("ENDPOINT_URL"))
print(os.getenv("DEPLOYMENT_NAME"))
print(os.getenv("API_VERSION"))