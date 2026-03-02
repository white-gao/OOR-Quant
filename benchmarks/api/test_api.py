from api import get_client_from_config

# Create client
client = get_client_from_config("gemini")

# Generate text
response = client.generate("Tell me a joke")
print(response)