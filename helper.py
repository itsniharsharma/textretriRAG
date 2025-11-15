def get_embedding(prompt, model ="nomic-embed-text"):

    import requests
    url = "http://localhost:11434/api/embeddings"
    data = {
        "model":model,
        "prompt":prompt,
        "max_tokens":1000,
        "stream":False,
        "temprature":0.7
    }
    response = requests.post(url, json=data)
    response.raise_for_status()

    return response.json().get("embedding", None)

def get_opensearch_client(host, port):
    from opensearchpy import OpenSearch

    client = OpenSearch(
        hosts = [{"host":host, "port":port}],
        http_compress = True,
        timeout=30,
        max_retries=3,
        retry_on_timeout=True,
    )

    if client.ping():
        print("Connected to OpenSeach")

    return client


if __name__ == "__main__":
    get_opensearch_client("localhost", 9200)