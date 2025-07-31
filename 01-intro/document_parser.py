import json

class DocumentParser:

    def __init__(self, documents_file: str):
        self.documents_file = documents_file

    @property
    def _documents(self):
        with open(self.documents_file, 'r') as file:
            documents = json.load(file)
        return documents

    @property
    def parsed_documents(self) -> list[dict[str, str]]:
        def add_course(document: dict[str, str], course: str) -> dict[str, str]:
            document["course"] = course
            return document
        
        output = []
        for document in self._documents:
            output.extend(list(map(lambda el: add_course(el, document["course"]), document["documents"])))
        return output
    
    
