from minsearch.minsearch import Index
from elasticsearch import Elasticsearch
from functools import cached_property
from tqdm.auto import tqdm
from typing import Protocol
from qdrant_client import QdrantClient, models


class SearchEngine(Protocol):
    def search(self, query: str, *args, **kwargs) -> list[dict[str, str]]:
        """Search method to be implemented by search engines."""
        pass


class MiniSearchEngine:
    def __init__(self, index: Index, documents: list[dict[str, str]]) -> None:
        self.index = index
        self.index.fit(documents)

    def search(self, 
            query: str, 
            boost_dict: dict[str, int | float], 
            filter_dict: dict[str, str], 
            num_result: int = 5
        ) -> list[dict[str, str]]:
        results = self.index.search(
            query=query,
            boost_dict=boost_dict,
            num_results=num_result,
            filter_dict=filter_dict
        )
        return results


class ElasticSearchEngine:
    def __init__(self, documents: list[dict[str, str]], index_name: str) -> None:
        self.documents = documents
        self.index_name = index_name
        self._create_index()

    @cached_property
    def es_client(self):
        return Elasticsearch("http://localhost:9200")
    
    @cached_property
    def index_settings(self):
        """can be moved to config class / configuration from yaml"""
        return {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "question": {"type": "text"},
                    "text": {"type": "text"},
                    "section": {"type": "keyword"},
                    "course": {"type": "keyword"}
                }
            }
        }
    
    def _create_index(self):
        if self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.delete(index=self.index_name)
        
        self.es_client.indices.create(index=self.index_name, body=self.index_settings)
        
        for doc in tqdm(self.documents):
            self.es_client.index(index=self.index_name, document=doc)

    def search(self, query: str) -> list[dict[str, str]]:
        response = self.es_client.search(index=self.index_name, body=query)
        return [hit['_source'] for hit in response['hits']['hits']]


class VectorSearchEngine:

    EMBEDDING_DIMENSIONALITY = 512

    def __init__(self, documents: list[dict[str, str]], model_handle: str, collection_name: str) -> None:
        self.documents = documents
        self.model_handle = model_handle
        self.collection_name = collection_name

    @cached_property
    def qdrant_client(self):
        return QdrantClient("http://localhost:6333")
    
    def crete_collection(self):
        self.qdrant_client.delete_collection(self.collection_name)
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "size": self.EMBEDDING_DIMENSIONALITY, 
                "distance": models.Distance.COSINE
            }
        )

        points = []
        for id, doc in enumerate(tqdm(self.documents)):
            text = doc["question"] + " " + doc["text"]
            vector = models.Document(text=text, model=self.model_handle)
            point = models.PointStruct(
                id=id,
                vector=vector,
                payload=doc
            )
            points.append(point)

        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def create_payload_index(self, field_name: str):
        """Create an index for the payload fields in the collection."""
        self.qdrant_client.create_payload_index(
            collection_name=self.collection_name,
            field_name=field_name,
            field_schema="keyword"
        )

    def search(self, query: str, filter_field: str, filter_value: str, num_result: int = 1):

        results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=models.Document( #embed the query text locally with "jinaai/jina-embeddings-v2-small-en"
                text=query,
                model=self.model_handle 
            ),
            query_filter=models.Filter( # filter by course name
            must=[
                models.FieldCondition(
                    key=filter_field,
                    match=models.MatchValue(value=filter_value)
                )
            ]
            ),
            limit=num_result, # top closest matches
            with_payload=True #to get metadata in the results
        )

        return [el.payload for el in results.points]