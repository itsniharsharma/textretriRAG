def create_index_if_not_exists(client, index_name):
   
   if client.indices.exists(index = index_name):
         print(f"Index '{index_name}' already exists.")
         #client.indices.delete(index_name)
    
   mappings = {
         "mappings": {
              "properties":{
                   "content": {"type": "text"},
                   "content_type": {"type": "keyword"},
                   "filename": {"type": "keyword"},
                   "embedding": {
                        "type": "dense_vector",
                        "dims": 768
                    },
              }
         },

         "settings":{
              "index":{
                   "knn": True,
                   "knn.space_type":"cosinesimil",

              }
         }
    }
   
   try:
        client.indices.create(index=index_name, body=mappings)
        print(f"Index '{index_name}' created successfully.")
   except Exception as e:
        print(f"Error creating index '{index_name}': {e}")
        raise
   

def prepare_chunks_for_ingstion(chunks):
     
     from helper import get_embedding
     prepared_chunks = []

     for idx, chunk in enumerate(chunks):
          if not chunk.get("content"):
               print(f"skipping empty chunk at index {idx}")
               continue
          
          chunk["embedding"] = get_embedding(chunk["content"])

          chunk_data = {
               "content":chunk.get("content", ""),
               "content_type":chunk.get("content_type", "text"),
               "filename":chunk.get("filename", None),
               "embedding":chunk.get("embedding", None),
          }
          prepared_chunks.append(chunk_data)

     return prepared_chunks

def ingest_chunks_into_opensearch(client, index_name, chunks):
     
     from opensearchpy import helpers

     actions = []
     for chunk in chunks:
          action = {
               "_index": index_name,
               "_source": chunk,
          }
          actions.append(action)

     try:
          helpers.bulk(client, actions)
          print(f"Successfully ingested {len(actions)} chunks into index '{index_name}'.")
     except Exception as e:
          print(f"Error ingesting chunks into index '{index_name}': {e}")
          raise
          

def ingest_all_content_into_opensearch(process_image, processed_table, sementic_chunks, index_name):
        from helper import get_opensearch_client
    
        client = get_opensearch_client("localhost", 9200)

        create_index_if_not_exists(client, index_name)
        image_chunks = prepare_chunks_for_ingstion(process_image)
        ingest_chunks_into_opensearch(client, index_name, image_chunks)

        table_chunks = prepare_chunks_for_ingstion(processed_table)
        ingest_chunks_into_opensearch(client, index_name, table_chunks)

        semantic_chunks_data = prepare_chunks_for_ingstion(sementic_chunks)
        ingest_chunks_into_opensearch(client, index_name, semantic_chunks_data)


if __name__ == "__main__":
    from unstructured.partition.pdf import partition_pdf
    from chunking import process_images_with_captions, process_tables_with_descriptions, create_semantic_chunks

    pdf_file_path = "files/my_paper.pdf"

    raw_chunks = partition_pdf(
     filename = pdf_file_path,
     strategy = "hi_res",
     infer_table_structure = True,
     extract_image_block_types = ["Image", "Figure", "Table"],
     extract_image_block_to_payload = True,
     chunking_strategy = None,
   )
    
    processed_image = process_images_with_captions(raw_chunks, use_gemini=True)

   
    process_tables = process_tables_with_descriptions(raw_chunks, use_gemini=True, use_ollama=False)
    # for table in process_tables:
    #     print(table)
    
    text_chunks = partition_pdf(
        filename = pdf_file_path,
        strategy = "hi_res",
        chunking_startegy = "by_title",
        max_characters = 2000,
        combine_text_under_n_chars = 500,
        new_after_n_chars = 1500,
    )

    sementic_chunks = create_semantic_chunks(text_chunks)

    index_name = "pdf_content"
    ingest_all_content_into_opensearch(processed_image, process_tables, sementic_chunks, index_name)