
from prompt_template import PromptTemplate
from document_parser import DocumentParser
from search_engine import VectorSearchEngine
from llm import Llm

MODEL = 'llama3.1:8b'
q = "how many zoomcamp is enrolled in a year?"

def rag(query, create_collection: bool = False) -> str:
    document_parser = DocumentParser("data/documents.json")
    engine = VectorSearchEngine(
        documents=document_parser.parsed_documents,
        model_handle="jinaai/jina-embeddings-v2-small-en",
        collection_name="zoomcamp-rag",
    )
    if create_collection:
        engine.crete_collection()
    engine.create_payload_index(field_name="course")
    results = engine.search(query=query, filter_field="course", filter_value="data-engineering-zoomcamp", num_result=5)

    prompt_template = PromptTemplate(
        query=query,
        search_result=results
    )

    llm = Llm(model=MODEL)
    response = llm.get_chat_esponse(prompt_template=str(prompt_template))
    
    return response

print(rag(q)) # Example usage of the rag function