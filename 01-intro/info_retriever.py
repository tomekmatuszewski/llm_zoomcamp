import json
from typing import Protocol, Any


class Indexable(Protocol):

    def __init__(self, text_fields: list[str], keyword_fields: list[str]):
        """Initialize the index with text and keyword fields."""
        pass

    def fit(self, docs: Any) -> dict[str, str]:
        """Method to convert the object to a dictionary representation."""
        pass

    def search(self, query: str, boost_dict: dict[str, float], num_results: int, filter_dict: dict[str, str]) -> list[dict[str, str]]:
        """Method to search the index."""
        pass


class InfoRetriever:
    def __init__(self, documents_file: str, index: Indexable):
        self.documents_file = documents_file
        self.index = index
        self.index.fit(self.parsed_documents)

    @property
    def documents(self):
        with open(self.documents_file, 'r') as file:
            documents = json.load(file)
        return documents

    @property
    def parsed_documents(self) -> list[dict[str, str]]:
        def add_course(document: dict[str, str], course: str) -> dict[str, str]:
            document["course"] = course
            return document
        
        output = []
        for document in self.documents:
            output.extend(list(map(lambda el: add_course(el, document["course"]), document["documents"])))
        return output
    
    def retrieve(self, query: str, boost_dict: dict[str, int | float], filter_dict: dict[str, str], num_result: int = 5) -> list[dict[str, str]]:
        results = self.index.search(
            query=query,
            boost_dict=boost_dict,
            num_results=num_result,
            filter_dict=filter_dict
        )
        return results
