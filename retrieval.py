# from helper import get_embedding, get_opensearch_client


# def keyword_search(query_text, top_k=20):
#     """
#     Perform keyword search using OpenSearch.

#     Args:
#         query_text (str): The query text to search for
#         top_k (int): Number of results to return

#     Returns:
#         list: Search results
#     """
#     client = get_opensearch_client("localhost", 9200)
#     index_name = "pdf_content"

#     try:
#         # Create a keyword search query
#         search_query = {
#             "size": top_k,
#             "query": {"match": {"content": query_text}},
#             "_source": ["content", "content_type", "token_count"],
#         }

#         response = client.search(index=index_name, body=search_query)
#         return response["hits"]["hits"]
#     except Exception as e:
#         print(f"Keyword search error: {e}")
#         return []


# def semantic_search(query_text, top_k=20):
#     """
#     Perform semantic search using vector embeddings.

#     Args:
#         query_text (str): The query text to search for
#         top_k (int): Number of results to return

#     Returns:
#         list: Search results
#     """
#     client = get_opensearch_client("localhost", 9200)
#     index_name = "localrag"

#     try:
#         # Get embedding for the query
#         query_embedding = get_embedding(query_text)

#         # Create a semantic search query
#         search_query = {
#             "size": top_k,
#             "query": {
#                 "knn": {
#                     "embedding": {
#                         "vector": query_embedding,
#                         "k": top_k,
#                     }
#                 }
#             },
#             "_source": ["content", "content_type", "token_count"],
#         }

#         response = client.search(index=index_name, body=search_query)
#         return response["hits"]["hits"]
#     except Exception as e:
#         print(f"Semantic search error: {e}")
#         return []


# def hybrid_search(query_text, top_k=20):
#     """
#     Perform hybrid search using both keyword and semantic search.

#     Args:
#         query_text (str): The query text to search for
#         top_k (int): Number of results to return

#     Returns:
#         list: Search results
#     """
#     client = get_opensearch_client("localhost", 9200)
#     index_name = "localrag"

#     try:
#         # Get embedding for the query
#         query_embedding = get_embedding(query_text)

#         # Create a hybrid search query
#         search_query = {
#             "size": top_k,
#             "query": {
#                 "bool": {
#                     "should": [
#                         {"knn": {"embedding": {"vector": query_embedding, "k": top_k}}},
#                         {"match": {"content": query_text}},
#                     ]
#                 }
#             },
#             "_source": ["content", "content_type", "token_count"],
#         }

#         response = client.search(index=index_name, body=search_query)
#         return response["hits"]["hits"]
#     except Exception as e:
#         print(f"Hybrid search error: {e}")
#         # Fall back to keyword search
#         try:
#             fallback_query = {
#                 "size": top_k,
#                 "query": {"match": {"content": query_text}},
#                 "_source": ["content", "content_type", "token_count"],
#             }
#             response = client.search(index=index_name, body=fallback_query)
#             return response["hits"]["hits"]
#         except Exception as e2:
#             print(f"Fallback search error: {e2}")
#             return []


# if __name__ == "__main__":
#     from pprint import pprint

#     query = "Compare RAG v/s fine-tuning"
#     results = keyword_search(query, top_k=10)
#     # results = semantic_search(query, top_k=10)
#     # results = hybrid_search(query, top_k=10)
#     pprint(results)



# from helper import get_embedding, get_opensearch_client


# INDEX_NAME = "pdf_content"     # <-- FIXED (same index as ingestion)


# # ---------------------------------------------------------
# # KEYWORD SEARCH
# # ---------------------------------------------------------
# def keyword_search(query_text, top_k=20):
#     client = get_opensearch_client("localhost", 9200)

#     query = {
#         "size": top_k,
#         "query": {
#             "match": {
#                 "content": {
#                     "query": query_text,
#                     "operator": "or"
#                 }
#             }
#         },
#         "_source": ["content", "content_type", "filename", "metadata"]
#     }

#     try:
#         resp = client.search(index=INDEX_NAME, body=query)
#         return resp["hits"]["hits"]
#     except Exception as e:
#         print(f"Keyword search error: {e}")
#         return []


# # ---------------------------------------------------------
# # SEMANTIC SEARCH (k-NN)
# # ---------------------------------------------------------
# def semantic_search(query_text, top_k=20):
#     client = get_opensearch_client("localhost", 9200)

#     # Generate embedding
#     query_vector = get_embedding(query_text)

#     query = {
#         "size": top_k,
#         "query": {
#             "knn": {
#                 "field": "embedding",
#                 "query_vector": query_vector,
#                 "k": top_k,
#                 "num_candidates": top_k
#             }
#         },
#         "_source": ["content", "content_type", "filename", "metadata"]
#     }

#     try:
#         resp = client.search(index=INDEX_NAME, body=query)
#         return resp["hits"]["hits"]
#     except Exception as e:
#         print(f"Semantic search error: {e}")
#         return []


# # ---------------------------------------------------------
# # HYBRID SEARCH (keyword + semantic)
# # ---------------------------------------------------------
# def hybrid_search(query_text, top_k=20):
#     client = get_opensearch_client("localhost", 9200)

#     query_vector = get_embedding(query_text)

#     query = {
#         "size": top_k,
#         "query": {
#             "bool": {
#                 "should": [
#                     {
#                         "knn": {
#                             "field": "embedding",
#                             "query_vector": query_vector,
#                             "k": top_k,
#                             "num_candidates": top_k
#                         }
#                     },
#                     {
#                         "match": {
#                             "content": {
#                                 "query": query_text,
#                                 "operator": "or"
#                             }
#                         }
#                     }
#                 ]
#             }
#         },
#         "_source": ["content", "content_type", "filename", "metadata"]
#     }

#     try:
#         resp = client.search(index=INDEX_NAME, body=query)
#         return resp["hits"]["hits"]
#     except Exception as e:
#         print(f"Hybrid search failed: {e}")
#         print("Switching to fallback keyword search...")
#         return keyword_search(query_text, top_k)


# # ---------------------------------------------------------
# # Manual test
# # ---------------------------------------------------------
# if __name__ == "__main__":
#     from pprint import pprint

#     query = "Explain RAG architecture"

#     print("\nKeyword Search:")
#     pprint(keyword_search(query, top_k=5))

#     print("\nSemantic Search:")
#     pprint(semantic_search(query, top_k=5))

#     print("\nHybrid Search:")
#     pprint(hybrid_search(query, top_k=5))


from helper import get_embedding, get_opensearch_client

# Your vector index name
INDEX_NAME = "pdf_content"


# ---------------------------------------------------------
# KEYWORD SEARCH
# ---------------------------------------------------------
def keyword_search(query_text, top_k=20):
    client = get_opensearch_client("localhost", 9200)

    query = {
        "size": top_k,
        "query": {
            "match": {
                "content": {
                    "query": query_text,
                    "operator": "or"
                }
            }
        },
        "_source": ["content", "content_type", "filename", "metadata"]
    }

    try:
        resp = client.search(index=INDEX_NAME, body=query)
        return resp["hits"]["hits"]
    except Exception as e:
        print(f"Keyword search error: {e}")
        return []


# ---------------------------------------------------------
# SEMANTIC SEARCH (k-NN)
# ---------------------------------------------------------
def semantic_search(query_text, top_k=20):
    client = get_opensearch_client("localhost", 9200)
    query_vector = get_embedding(query_text)

    if query_vector is None:
        print("Embedding failed, fallback to keyword search")
        return keyword_search(query_text, top_k)

    # Correct OpenSearch KNN format
    query = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {           
                    "vector": query_vector,
                    "k": top_k
                }
            }
        },
        "_source": ["content", "content_type", "filename", "metadata"]
    }

    try:
        resp = client.search(index=INDEX_NAME, body=query)
        return resp["hits"]["hits"]
    except Exception as e:
        print(f"Semantic search error: {e}")
        return []


# ---------------------------------------------------------
# HYBRID SEARCH (keyword + semantic)
# ---------------------------------------------------------
def hybrid_search(query_text, top_k=20):
    client = get_opensearch_client("localhost", 9200)
    query_vector = get_embedding(query_text)

    if query_vector is None:
        print("Embedding failed, using keyword-only fallback")
        return keyword_search(query_text, top_k)

    # Correct hybrid query using OpenSearch FAISS KNN syntax
    query = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "embedding": {      
                                "vector": query_vector,
                                "k": top_k
                            }
                        }
                    },
                    {
                        "match": {
                            "content": {
                                "query": query_text,
                                "operator": "or"
                            }
                        }
                    }
                ]
            }
        },
        "_source": ["content", "content_type", "filename", "metadata"]
    }

    try:
        resp = client.search(index=INDEX_NAME, body=query)
        return resp["hits"]["hits"]
    except Exception as e:
        print(f"Hybrid search failed: {e}")
        print("Switching to fallback keyword search...")
        return keyword_search(query_text, top_k)


# ---------------------------------------------------------
# Manual test
# ---------------------------------------------------------
if __name__ == "__main__":
    from pprint import pprint
    query = "Explain RAG architecture"

    print("\nKeyword Search:")
    pprint(keyword_search(query, top_k=5))

    print("\nSemantic Search:")
    pprint(semantic_search(query, top_k=5))

    print("\nHybrid Search:")
    pprint(hybrid_search(query, top_k=5))
