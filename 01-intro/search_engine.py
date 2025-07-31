from minsearch.minsearch import Index
from elasticsearch import Elasticsearch
from functools import cached_property
from tqdm.auto import tqdm
from typing import Protocol


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