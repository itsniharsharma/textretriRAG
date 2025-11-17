# import json
# import os

# import google.generativeai as genai
# import requests
# from dotenv import load_dotenv
# from langchain_core.prompts import PromptTemplate

# # Import retrieval functions
# from retrieval import hybrid_search, keyword_search, semantic_search

# # Load environment variables
# load_dotenv()

# # Configure Gemini API with validation
# gemini_api_key = os.getenv("GEMINI_API_KEY")
# if not gemini_api_key:
#     print("ERROR: GEMINI_API_KEY not found in environment variables")
# else:
#     print(f"Configuring Gemini with API key: {gemini_api_key[:5]}...")
#     genai.configure(api_key=gemini_api_key)

# # Define RAG prompt template
# RAG_PROMPT_TEMPLATE = """
# You are an AI assistant helping answer questions about Retrieval-Augmented Generation (RAG).
# Use the following retrieved documents to answer the user's question.
# If the retrieved documents don't contain relevant information, say that you don't know.

# RETRIEVED DOCUMENTS:
# {context}

# USER QUESTION:
# {question}

# YOUR ANSWER (be comprehensive, accurate, and helpful):
# """

# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=RAG_PROMPT_TEMPLATE,
# )


# def generate_with_gemini(prompt_text, model_name="gemini-1.5-flash", stream=False):
#     """Generate response using Google's Gemini model with robust error handling"""
#     try:
#         # 1. Initialize model
#         print(f"Initializing Gemini model: {model_name}")
#         model = genai.GenerativeModel(model_name)

#         # 2. Safety check for prompt length
#         if len(prompt_text) > 30000:
#             prompt_text = prompt_text[:30000] + "...[truncated due to length]"
#             print(f"Warning: Prompt was truncated to 30000 characters")

#         # 3. Set up generation configuration
#         generation_config = {
#             "temperature": 0.7,
#             "top_p": 0.95,
#             "top_k": 40,
#             "max_output_tokens": 2048,
#         }

#         # 4. Configure safety settings to prevent blocking
#         safety_settings = {
#             "harassment": "block_none",
#             "hate": "block_none",
#             "sexual": "block_none",
#             "dangerous": "block_none",
#         }

#         # 5. Handle streaming vs non-streaming differently
#         if stream:
#             print("Starting streaming response generation...")
#             response_generator = model.generate_content(
#                 contents=prompt_text,
#                 generation_config=generation_config,
#                 safety_settings=safety_settings,
#                 stream=True,
#             )

#             # Process stream chunks correctly
#             for chunk in response_generator:
#                 # Most direct way to get text from a chunk
#                 if hasattr(chunk, "text"):
#                     if chunk.text:  # Only yield non-empty text
#                         yield chunk.text
#                 # Alternative way through parts
#                 elif hasattr(chunk, "parts"):
#                     for part in chunk.parts:
#                         if hasattr(part, "text") and part.text:
#                             yield part.text
#         else:
#             print("Requesting non-streaming response...")
#             response = model.generate_content(
#                 contents=prompt_text,
#                 generation_config=generation_config,
#                 safety_settings=safety_settings,
#             )

#             # Extract text from response
#             if hasattr(response, "text"):
#                 return response.text
#             elif hasattr(response, "parts") and response.parts:
#                 return "".join([p.text for p in response.parts if hasattr(p, "text")])
#             else:
#                 return f"Response received but couldn't extract text: {str(response)}"

#     except Exception as e:
#         import traceback

#         error_msg = f"Error with Gemini generation: {str(e)}\n{traceback.format_exc()}"
#         print(error_msg)
#         if stream:
#             yield error_msg
#         else:
#             return error_msg


# def generate_with_ollama(prompt_text, model_name="deepseek-r1:1.5b", stream=False):
#     """Generate response using Ollama with Deepseek model"""
#     try:
#         url = "http://localhost:11434/api/generate"
#         data = {
#             "model": model_name,
#             "prompt": prompt_text,
#             "stream": stream,
#             "options": {"temperature": 0.7},
#         }

#         if stream:
#             response = requests.post(url, json=data, stream=True)
#             response.raise_for_status()

#             for line in response.iter_lines():
#                 if line:
#                     try:
#                         chunk = json.loads(line.decode("utf-8"))
#                         if "response" in chunk:
#                             yield chunk["response"]
#                     except json.JSONDecodeError:
#                         continue
#         else:
#             response = requests.post(url, json=data)
#             response.raise_for_status()
#             return response.json().get("response", "No response generated")
#     except Exception as e:
#         error_msg = f"Error generating response with Ollama: {str(e)}"
#         if stream:
#             yield error_msg
#         else:
#             return error_msg


# def generate_rag_response(
#     query, search_type="hybrid", top_k=5, model_type="gemini", stream=False
# ):
#     """
#     Generate RAG response using retrieved chunks.

#     Args:
#         query: User query
#         search_type: Type of search (keyword, semantic, hybrid)
#         top_k: Number of chunks to retrieve
#         model_type: Type of model to use (gemini, ollama)
#         stream: Whether to stream the response

#     Returns:
#         Generated response or generator for streaming
#     """
#     try:
#         # Step 1: Retrieve relevant chunks based on search type
#         if search_type == "keyword":
#             results = keyword_search(query, top_k=top_k)
#         elif search_type == "semantic":
#             results = semantic_search(query, top_k=top_k)
#         else:  # hybrid
#             results = hybrid_search(query, top_k=top_k)

#         if not results:
#             message = "No relevant information found. Please try a different search type or refine your question."
#             if stream:
#                 yield message
#                 return
#             else:
#                 return message

#         # Step 2: Format retrieved contexts
#         contexts = []
#         for i, hit in enumerate(results):
#             source = hit["_source"]
#             content = source.get("content", "")
#             content_type = source.get("content_type", "unknown")

#             # Add metadata if available
#             metadata_info = ""
#             if "metadata" in source and source["metadata"]:
#                 if "caption" in source["metadata"] and source["metadata"]["caption"]:
#                     metadata_info += f"\nCaption: {source['metadata']['caption']}"

#             context_entry = (
#                 f"[Document {i+1} - {content_type}]{metadata_info}\n{content}"
#             )
#             contexts.append(context_entry)

#         # Step 3: Format the prompt using LangChain template
#         context_text = "\n\n---\n\n".join(contexts)
#         prompt_text = prompt.format(context=context_text, question=query)

#         # Step 4: Generate response with selected model
#         if model_type == "gemini":
#             if stream:
#                 yield from generate_with_gemini(prompt_text, stream=True)
#             else:
#                 return generate_with_gemini(prompt_text, stream=False)
#         else:  # ollama
#             if stream:
#                 yield from generate_with_ollama(prompt_text, stream=True)
#             else:
#                 return generate_with_ollama(prompt_text, stream=False)

#     except Exception as e:
#         error_message = f"Error in RAG process: {str(e)}"
#         if stream:
#             yield error_message
#         else:
#             return error_message


# # For testing
# if __name__ == "__main__":
#     # Test both streaming and non-streaming
#     query = "How does RAG work?"

#     # Test streaming
#     print("Response: ", end="", flush=True)
#     for chunk in generate_rag_response(query, "hybrid", 3, "gemini", True):
#         print(chunk, end="", flush=True)

#     # Generating streaming response with Ollama
#     print("\n\nOllama Streaming Response: ", end="", flush=True)
#     for chunk in generate_rag_response(query, "hybrid", 3, "ollama", True):
#         print(chunk, end="", flush=True)



import json
import os
import google.generativeai as genai
import requests
from dotenv import load_dotenv

# LangChain v1.x imports (correct)
from langchain_core.prompts import PromptTemplate

# Retrieval functions
from retrieval import hybrid_search, keyword_search, semantic_search


# ------------------------------------------------------------
# Load environment variables and configure Gemini
# ------------------------------------------------------------
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("ERROR: GEMINI_API_KEY missing")
else:
    print(f"Configuring Gemini API (key starts with {gemini_api_key[:5]}...)")
    genai.configure(api_key=gemini_api_key)


# ------------------------------------------------------------
# RAG Prompt
# ------------------------------------------------------------
RAG_PROMPT_TEMPLATE = """
You are an AI assistant answering questions using Retrieval-Augmented Generation (RAG).
Use ONLY the retrieved documents provided below.

If the documents do not contain the answer, say:
"I’m not able to find relevant information in the retrieved documents."

------------------------------------
RETRIEVED DOCUMENTS:
{context}
------------------------------------

USER QUESTION:
{question}

Your Final Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


# ------------------------------------------------------------
# Gemini Generation
# ------------------------------------------------------------
def generate_with_gemini(prompt_text, model_name="gemini-2.0-flash", stream=False):
    try:
        print(f"Initializing Gemini model: {model_name}")
        model = genai.GenerativeModel(model_name)

        # Safe truncate long prompt
        if len(prompt_text) > 30000:
            print("Prompt too long; truncating")
            prompt_text = prompt_text[:30000]

        generation_config = {
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        if stream:
            print("Streaming Gemini response...")
            response_stream = model.generate_content(
                prompt_text,
                generation_config=generation_config,
                stream=True
            )

            for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, "parts"):
                    for p in chunk.parts:
                        if hasattr(p, "text") and p.text:
                            yield p.text

        else:
            print("Requesting non-streaming Gemini response...")
            response = model.generate_content(
                prompt_text,
                generation_config=generation_config,
            )

            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "parts"):
                return "".join([p.text for p in response.parts])
            else:
                return "Gemini returned an empty response."

    except Exception as e:
        error_msg = f"Gemini Error: {str(e)}"
        print(error_msg)
        return error_msg if not stream else (yield error_msg)


# ------------------------------------------------------------
# Ollama / DeepSeek Generation
# ------------------------------------------------------------
def generate_with_ollama(prompt_text, model_name="deepseek-r1:1.5b", stream=False):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "stream": stream,
            "options": {"temperature": 0.5},
        }

        if stream:
            response = requests.post(url, json=payload, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    if "response" in chunk:
                        yield chunk["response"]
                except:
                    continue

        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")

    except Exception as e:
        msg = f"Ollama Error: {str(e)}"
        print(msg)
        return msg if not stream else (yield msg)


# ------------------------------------------------------------
# RAG Response Generator
# ------------------------------------------------------------
def generate_rag_response(query, search_type="hybrid", top_k=5, model_type="gemini", stream=False):
    try:
        # Step 1 — Retrieve documents
        if search_type == "keyword":
            results = keyword_search(query, top_k=top_k)
        elif search_type == "semantic":
            results = semantic_search(query, top_k=top_k)
        else:
            results = hybrid_search(query, top_k=top_k)

        if not results:
            no_data_msg = "No relevant content found in the document store."
            return no_data_msg if not stream else (yield no_data_msg)

        # Step 2 — Format retrieved chunks
        contexts = []
        for i, hit in enumerate(results):
            src = hit["_source"]
            content = src.get("content", "")
            ctype = src.get("content_type", "unknown")

            caption = ""
            if "metadata" in src and src["metadata"]:
                caption = src["metadata"].get("caption", "")

            entry = f"[Document {i+1} - {ctype}]\n"
            if caption:
                entry += f"Caption: {caption}\n"
            entry += content

            contexts.append(entry)

        context_text = "\n\n---\n\n".join(contexts)

        # Step 3 — Final RAG prompt
        final_prompt = prompt.format(context=context_text, question=query)

        # Step 4 — Pick model
        if model_type == "gemini":
            if stream:
                yield from generate_with_gemini(final_prompt, stream=True)
            else:
                return generate_with_gemini(final_prompt)

        else:  # Ollama
            if stream:
                yield from generate_with_ollama(final_prompt, stream=True)
            else:
                return generate_with_ollama(final_prompt)

    except Exception as e:
        msg = f"Error in RAG pipeline: {str(e)}"
        return msg if not stream else (yield msg)


# ------------------------------------------------------------
# Manual test
# ------------------------------------------------------------
if __name__ == "__main__":
    print("Testing RAG pipeline...\n")
    query = "What is RAG and how does it work?"

    print("\nGemini Streaming:")
    for chunk in generate_rag_response(query, "hybrid", 3, "gemini", True):
        print(chunk, end="", flush=True)

    print("\n\nOllama Streaming:")
    for chunk in generate_rag_response(query, "hybrid", 3, "ollama", True):
        print(chunk, end="", flush=True)
