from minsearch.minsearch import Index
from prompt_template import PromptTemplate
from document_parser import DocumentParser
from search_engine import MiniSearchEngine
from llm import Llm

MODEL = 'llama3.1:8b'
q = "How do I run kafka?"

def rag(query):
    document_parser = DocumentParser("data/documents.json")
    index = Index(
        text_fields=["text", "question", "section"],
        keyword_fields=["course"],
    )

    search_engine = MiniSearchEngine(index=index, documents=document_parser.parsed_documents)
    results = search_engine.search(
        query=q,
        boost_dict={"question": 3, "section": 0.5},
        filter_dict={"course": "data-engineering-zoomcamp"},
        num_result=5
    )


    prompt_template = PromptTemplate(
        query=query,
        search_result=results
    )

    llm = Llm(model=MODEL)
    response = llm.get_chat_esponse(prompt_template=str(prompt_template))
    
    return response

print(rag(q))  # Example usage of the rag function