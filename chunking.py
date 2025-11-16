# def process_images_with_captions(raw_chunks, use_gemini=True):
#     import base64
#     import os
#     import google.generativeai as genai
#     from dotenv import load_dotenv
#     from unstructured.documents.elements import FigureCaption, Image

#     load_dotenv()

#     # Configure Gemini API
#     if use_gemini:
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
#         genai.configure(api_key=api_key, transport="rest")

#     processed_images = []
#     encountered_errors = []

#     for idx, chunk in enumerate(raw_chunks):
#         if isinstance(chunk, Image):

#             # Find caption
#             if idx + 1 < len(raw_chunks) and isinstance(raw_chunks[idx + 1], FigureCaption):
#                 caption = raw_chunks[idx + 1].text
#             else:
#                 caption = "No caption available"

#             image_data = {
#                 "caption": caption,
#                 "image_text": chunk.text if hasattr(chunk, "text") else "",
#                 "base64_image": chunk.metadata.image_base64,
#                 "content": chunk.text if hasattr(chunk, "text") else "",
#                 "content_type": "image",
#                 "filename": chunk.metadata.filename if hasattr(chunk, "metadata") else "",
#             }

#             error_data = {"error": None, "error_message": None}

#             # Generate description using Gemini
#             if use_gemini:
#                 try:
#                     image_binary = base64.b64decode(chunk.metadata.image_base64)

#                     model = genai.GenerativeModel("models/gemini-2.0-flash-image")

#                     prompt = (
#                         "Generate a comprehensive and detailed description of this image "
#                         "from a technical document about Retrieval-Augmented Generation (RAG).\n\n"
#                         f"CONTEXT INFORMATION:\n"
#                         f"- Caption: {caption}\n"
#                         f"- Text extracted from image: {chunk.text if hasattr(chunk, 'text') else 'No text'}\n\n"
#                         "DESCRIPTION REQUIREMENTS:\n"
#                         "1. Overview of the image (diagram/chart/architecture).\n"
#                         "2. Describe components and data flow.\n"
#                         "3. Explain axes or metrics if it’s a chart.\n"
#                         "4. Interpret abbreviations.\n"
#                         "5. Explain how this relates to RAG.\n"
#                         "6. Include metrics/trends.\n"
#                         "7. Length: 150-300 words.\n"
#                     )

#                     response = model.generate_content(
#                         contents=[prompt, {"mime_type": "image/jpeg", "data": image_binary}]
#                     )

#                     image_data["content"] = response.text

#                 except Exception as e:
#                     print(f"Warning: Error generating description: {str(e)}")
#                     error_data["error"] = str(e)
#                     error_data["error_message"] = "Error generating description with Gemini."
#                     encountered_errors.append(error_data)

#             processed_images.append(image_data)

#     print(f"Processed {len(processed_images)} images with captions and descriptions")
#     print(f"Errors encountered: {len(encountered_errors)}")

#     return processed_images, encountered_errors



# def process_tables_with_descriptions(raw_chunks, use_gemini=True, use_ollama=False):
#     """Process tables with descriptions using Gemini or Ollama"""
#     import os
#     import google.generativeai as genai
#     import requests
#     from dotenv import load_dotenv
#     from unstructured.documents.elements import Table

#     load_dotenv()

#     # Configure Gemini
#     if use_gemini:
#         api_key = os.getenv("GEMINI_API_KEY")
#         if not api_key:
#             raise ValueError("GEMINI_API_KEY is not set.")
#         genai.configure(api_key=api_key, transport="rest")

#     processed_tables = []
#     encountered_errors = []

#     for idx, chunk in enumerate(raw_chunks):
#         if isinstance(chunk, Table):

#             table_data = {
#                 "table_as_html": chunk.metadata.text_as_html,
#                 "table_text": chunk.text,
#                 "content": chunk.text,
#                 "content_type": "table",
#                 "filename": chunk.metadata.filename if hasattr(chunk, "metadata") else "",
#             }

#             # Gemini summary
#             if use_gemini:
#                 try:
#                     model = genai.GenerativeModel("models/gemini-2.5-flash")

#                     prompt = (
#                         "Generate a comprehensive and detailed description of the following table "
#                         "from a technical document about Retrieval-Augmented Generation (RAG).\n\n"
#                         f"TABLE HTML:\n{chunk.metadata.text_as_html}\n\n"
#                         "REQUIREMENTS:\n"
#                         "1. Explain purpose of the table.\n"
#                         "2. Explain columns/rows.\n"
#                         "3. Describe trends/insights.\n"
#                         "4. Relate to RAG.\n"
#                         "5. Length: 150–300 words.\n"
#                         "Do not write phrases like 'This table shows'. Direct explanation only."
#                     )

#                     response = model.generate_content(contents=[prompt])
#                     table_data["content"] = response.text

#                 except Exception as e:
#                     encountered_errors.append({
#                         "error": str(e),
#                         "error_message": "Error generating description with Gemini.",
#                     })

#             # Ollama fallback
#             elif use_ollama:
#                 try:
#                     url = "http://localhost:11434/api/generate"
#                     data = {
#                         "model": "deepseek-r1:1.5b",
#                         "prompt": (
#                             "Analyze the following table and provide a detailed summary of its contents.\n"
#                             f"HTML: {chunk.metadata.text_as_html}"
#                         ),
#                         "max_tokens": 1000,
#                         "stream": False,
#                         "temperature": 0.2,
#                     }

#                     response = requests.post(url, json=data)
#                     response.raise_for_status()

#                     table_data["content"] = response.json().get("response", "")

#                 except Exception as e:
#                     encountered_errors.append({
#                         "error": str(e),
#                         "error_message": "Error generating description with Ollama.",
#                     })

#             processed_tables.append(table_data)

#     print(f"Processed {len(processed_tables)} tables with descriptions")
#     print(f"Errors encountered: {len(encountered_errors)}")

#     return processed_tables, encountered_errors



# def create_semantic_chunks(chunks):
#     from unstructured.documents.elements import CompositeElement
#     processed_chunks = []

#     for idx, chunk in enumerate(chunks):
#         if isinstance(chunk, CompositeElement):
#             processed_chunks.append({
#                 "content": chunk.text,
#                 "content_type": "text",
#                 "filename": chunk.metadata.filename if hasattr(chunk, "metadata") else "",
#             })

#     print(f"Created {len(processed_chunks)} semantic chunks from document")
#     return processed_chunks



# if __name__ == "__main__":
#     from unstructured.partition.pdf import partition_pdf

#     pdf_file_path = "files/my_paper.pdf"

#     raw_chunks = partition_pdf(
#         filename=pdf_file_path,
#         strategy="hi_res",
#         infer_table_structure=True,
#         extract_image_block_types=["Image", "Figure", "Table"],
#         extract_image_block_to_payload=True,
#         chunking_strategy=None,
#     )

#     process_tables = process_tables_with_descriptions(raw_chunks, use_gemini=True)
#     for table in process_tables:
#         print(table)



import base64
import os
import google.generativeai as genai
from dotenv import load_dotenv
from unstructured.documents.elements import FigureCaption, Image, Table, CompositeElement


# ------------------------------------------------------------
# Load & Configure Gemini
# ------------------------------------------------------------
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")


# ------------------------------------------------------------
# IMAGE PROCESSING
# ------------------------------------------------------------
def process_images_with_captions(raw_chunks, use_gemini=True):
    """
    Extract images + captions + generate descriptions with Gemini.
    """

    processed_images = []
    encountered_errors = []

    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, Image):

            # Extract caption
            if idx + 1 < len(raw_chunks) and isinstance(raw_chunks[idx + 1], FigureCaption):
                caption = raw_chunks[idx + 1].text
            else:
                caption = "No caption available"

            image_data = {
                "caption": caption,
                "image_text": getattr(chunk, "text", ""),
                "base64_image": chunk.metadata.image_base64,
                "content": getattr(chunk, "text", ""),
                "content_type": "image",
                "filename": getattr(chunk.metadata, "filename", ""),
            }

            if use_gemini:
                try:
                    image_binary = base64.b64decode(chunk.metadata.image_base64)

                    # Correct image-compatible model
                    model = genai.GenerativeModel("models/gemini-1.5-flash")

                    prompt = (
                        "You are analyzing an image extracted from a technical document "
                        "related to Retrieval-Augmented Generation (RAG).\n\n"
                        f"Caption: {caption}\n"
                        f"Extracted text: {image_data['image_text']}\n\n"
                        "Provide a detailed technical description covering:\n"
                        "1. Diagram/chart/architecture overview.\n"
                        "2. Components & data flow.\n"
                        "3. Technical purpose & meaning.\n"
                        "4. Metrics or labels if present.\n"
                        "5. How this visual relates to RAG.\n"
                        "6. Write 150–300 words.\n"
                    )

                    response = model.generate_content(
                        [prompt, {"mime_type": "image/jpeg", "data": image_binary}]
                    )

                    image_data["content"] = response.text

                except Exception as e:
                    print(f"Error describing image: {e}")
                    encountered_errors.append({"error": str(e)})

            processed_images.append(image_data)

    print(f"Processed {len(processed_images)} images")
    if encountered_errors:
        print(f"Errors: {len(encountered_errors)}")

    return processed_images   


# ------------------------------------------------------------
# TABLE PROCESSING
# ------------------------------------------------------------
def process_tables_with_descriptions(raw_chunks, use_gemini=True, use_ollama=False):
    """
    Extract tables + generate summaries.
    """

    processed_tables = []
    encountered_errors = []

    for idx, chunk in enumerate(raw_chunks):
        if isinstance(chunk, Table):

            table_data = {
                "table_as_html": chunk.metadata.text_as_html,
                "table_text": chunk.text,
                "content": chunk.text,
                "content_type": "table",
                "filename": getattr(chunk.metadata, "filename", ""),
            }

            if use_gemini:
                try:
                    model = genai.GenerativeModel("models/gemini-1.5-flash")

                    prompt = (
                        "Analyze the following table extracted from a technical RAG document.\n\n"
                        f"{chunk.metadata.text_as_html}\n\n"
                        "Explain:\n"
                        "1. Purpose of the table\n"
                        "2. Meaning of each row/column\n"
                        "3. Patterns, comparisons, or metrics\n"
                        "4. Relation to RAG concepts\n"
                        "Write 150–300 words.\n"
                        "Do NOT start with 'This table shows'."
                    )

                    response = model.generate_content(prompt)
                    table_data["content"] = response.text

                except Exception as e:
                    print(f"Table description error: {e}")
                    encountered_errors.append({"error": str(e)})

            elif use_ollama:
                try:
                    import requests

                    url = "http://localhost:11434/api/generate"
                    body = {
                        "model": "deepseek-r1:1.5b",
                        "prompt": (
                            "Analyze this table and summarize its purpose and contents:\n"
                            f"{chunk.metadata.text_as_html}"
                        ),
                        "stream": False
                    }

                    r = requests.post(url, json=body)
                    r.raise_for_status()
                    table_data["content"] = r.json().get("response", "")

                except Exception as e:
                    encountered_errors.append({"error": str(e)})

            processed_tables.append(table_data)

    print(f"Processed {len(processed_tables)} tables")
    if encountered_errors:
        print(f"Errors: {len(encountered_errors)}")

    return processed_tables   # FIXED (not tuple)


# ------------------------------------------------------------
# TEXT CHUNKING
# ------------------------------------------------------------
def create_semantic_chunks(chunks):
    processed = []

    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, CompositeElement):
            processed.append({
                "content": chunk.text,
                "content_type": "text",
                "filename": getattr(chunk.metadata, "filename", "")
            })

    print(f"Created {len(processed)} text chunks")
    return processed


# ------------------------------------------------------------
# Debug mode
# ------------------------------------------------------------
if __name__ == "__main__":
    from unstructured.partition.pdf import partition_pdf

    pdf = "files/my_paper.pdf"

    raw = partition_pdf(
        filename=pdf,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Figure", "Table"],
        extract_image_block_to_payload=True
    )

    tables = process_tables_with_descriptions(raw, use_gemini=True)
    print(tables)

