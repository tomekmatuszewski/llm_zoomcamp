
from prompt_template import PromptTemplate
from document_parser import DocumentParser
from search_engine import ElasticSearchEngine
from llm import Llm

MODEL = 'llama3.1:8b'
q = "how many zoomcamp is enrolled in a year?"

def rag(query):
    document_parser = DocumentParser("data/documents.json")
    index_name = "course-questions"
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }

    elastic_search_engine = ElasticSearchEngine(document_parser.parsed_documents, index_name=index_name)
    results = elastic_search_engine.search(search_query)



    prompt_template = PromptTemplate(
        query=query,
        search_result=results
    )

    llm = Llm(model=MODEL)
    response = llm.get_chat_esponse(prompt_template=str(prompt_template))
    
    return response

print(rag(q)) # Example usage of the rag function