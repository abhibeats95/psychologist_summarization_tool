nice_class_schema={'class': '',
 'invertedIndexConfig': {'bm25': {'b': 0.75, 'k1': 1.2},
  'cleanupIntervalSeconds': 60,
  'indexPropertyLength': True,
  'stopwords': {'additions': None, 'preset': 'en', 'removals': None}},
 'moduleConfig': {'text2vec-openai': {'model': 'ada','modelVersion': '002',
   'type': 'text',
   'vectorizeClassName': False}},
 'multiTenancyConfig': {'enabled': False},
 'properties': [{'dataType': ['text'],
   'indexFilterable': True,
   'indexSearchable': True,
   'moduleConfig': {'text2vec-openai': {'skip': False,
     'vectorizePropertyName': True}},
   'name': 'title',
   'tokenization': 'word'},
  {'dataType': ['text'],
   'indexFilterable': True,
   'indexSearchable': True,
   'moduleConfig': {'text2vec-openai': {'skip': False,
     'vectorizePropertyName': True}},
   'name': 'content',
   'tokenization': 'word'}],
 'replicationConfig': {'factor': 1},
 'shardingConfig': {'virtualPerPhysical': 128,
  'desiredCount': 2,
  'actualCount': 2,
  'desiredVirtualCount': 256,
  'actualVirtualCount': 256,
  'key': '_id',
  'strategy': 'hash',
  'function': 'murmur3'},
 'vectorIndexConfig': {'skip': False,
  'cleanupIntervalSeconds': 300,
  'maxConnections': 64,
  'efConstruction': 128,
  'ef': -1,
  'dynamicEfMin': 100,
  'dynamicEfMax': 500,
  'dynamicEfFactor': 8,
  'vectorCacheMaxObjects': 1000000000000,
  'flatSearchCutoff': 40000,
  'distance': 'cosine',
  'pq': {'enabled': False,
   'bitCompression': False,
   'segments': 0,
   'centroids': 256,
   'trainingLimit': 100000,
   'encoder': {'type': 'kmeans', 'distribution': 'log-normal'}}},
 'vectorIndexType': 'hnsw',
 'vectorizer': 'text2vec-openai'}



from typing import Union
from langchain_core.documents.base import Document
from time import sleep

def get_failed_objects(results, error_logs, failed_docs):
    if results is None:
        return None
    
    for result in results:
        if result.get('result', {}).get('status') == 'SUCCESS':
            continue

        error_logs.append(result["result"])
        
        content = result.get('properties', {}).pop('content', None)
        meta = result.get('properties', {})
        
        if content is not None:
            doc = Document(page_content=content, metadata=meta)
            failed_docs.append(doc)


def create_and_embed_objects(client,class_name, documents,sleep_seconds):
    # Initialize lists for storing errors and failed document processing results
    failed_docs = []
    error_logs = []

    # Pass failed_docs and error_logs to the callback function
    callback_function = lambda results: get_failed_objects(results, error_logs, failed_docs)

    client.batch.configure(batch_size=50, dynamic=True, creation_time=5, timeout_retries=3, connection_error_retries=3, callback=callback_function)
    with client.batch as batch:
        for i, doc in enumerate(documents):
            if isinstance(doc, Document):
                weaviate_object = doc.metadata
                weaviate_object["content"] = doc.page_content
            else:
                weaviate_object = doc
                print("Metadata didn't load!")

            try:
                obj_id = batch.add_data_object(data_object=weaviate_object, class_name=class_name)
            except:
                print('Failed to add object')

            
            # Sleep after every 1000 documents, but not at the very first document
            if (i + 1) % 1000 == 0 and i != 0:
                print(f"Processed {i+1} documents, pausing for {sleep_seconds} seconds...")
                sleep(sleep_seconds)
    return failed_docs, error_logs



