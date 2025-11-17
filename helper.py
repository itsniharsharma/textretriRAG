# def get_embedding(prompt, model ="nomic-embed-text"):

#     import requests
#     url = "http://localhost:11434/api/embeddings"
#     data = {
#         "model":model,
#         "prompt":prompt,
#         "max_tokens":1000,
#         "stream":False,
#         "temprature":0.7
#     }
#     response = requests.post(url, json=data)
#     response.raise_for_status()

#     return response.json().get("embedding", None)

# def get_opensearch_client(host, port):
#     from opensearchpy import OpenSearch

#     client = OpenSearch(
#         hosts = [{"host":host, "port":port}],
#         http_compress = True,
#         timeout=30,
#         max_retries=3,
#         retry_on_timeout=True,
#     )

#     if client.ping():
#         print("Connected to OpenSeach")

#     return client


# if __name__ == "__main__":
#     get_opensearch_client("localhost", 9200)


import requests
import numpy as np
from opensearchpy import OpenSearch


# ----------------------------------------------------------
# Generate Embeddings via Ollama
# ----------------------------------------------------------
def get_embedding(prompt, model="nomic-embed-text"):
    """
    Generate text embeddings using Ollama.
    Ensures correct format, error handling, and float32 output.
    """

    url = "http://localhost:11434/api/embeddings"

    payload = {
        "model": model,
        "prompt": prompt
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code != 200:
            print(f"Ollama embedding error: {response.text}")
            return None

        output = response.json()
        embedding = output.get("embedding")
        # embedding = np.array(embedding, dtype=np.float32).tolist()

        if embedding is None:
            print("Failed to get embedding from Ollama response.")
            return None

        # Convert to float32 numpy array (recommended for OpenSearch)
        embedding = np.array(embedding, dtype=np.float32).tolist()
        return embedding

    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


# ----------------------------------------------------------
# Create OpenSearch Client
# ----------------------------------------------------------
def get_opensearch_client(host="localhost", port=9200):
    """
    Initialize OpenSearch client with reliable settings.
    """

    client = OpenSearch(
        hosts=[{"host": host, "port": port}],
        http_compress=True,
        timeout=60,
        max_retries=5,
        retry_on_timeout=True,
        scheme="http",                       # important for Windows local
        verify_certs=False,                  # disable SSL for local OS
        ssl_show_warn=False,
    )

    try:
        if client.ping():
            print("Connected to OpenSearch")
        else:
            print("Failed to connect to OpenSearch")
    except Exception as e:
        print(f"Error connecting to OpenSearch: {e}")

    return client


# ----------------------------------------------------------
# Testing
# ----------------------------------------------------------
if __name__ == "__main__":
    print("Testing OpenSearch Connection...")
    get_opensearch_client()

    print("\nTesting Embedding...")
    emb = get_embedding("Hello world")
    print(f"Embedding length: {len(emb) if emb else 'None'}")
