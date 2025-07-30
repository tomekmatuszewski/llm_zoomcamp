from minsearch.minsearch import Index
from prompt_template import PromptTemplate
from info_retriever import InfoRetriever
from llm import Llm


MODEL = 'llama3.1:8b'
q = "How do I run kafka?"

index = Index(
    text_fields=["text", "question", "section"],
    keyword_fields=["course"],
)

retriever = InfoRetriever(
    documents_file="data/documents.json",
    index=index
)

results = retriever.retrieve(
    query=q,
    boost_dict={"question": 3, "section": 0.5},
    filter_dict={"course": "data-engineering-zoomcamp"},
    num_result=5
)

prompe_template = PromptTemplate(
    query=q,
    search_result=results
)

llm = Llm(model=MODEL)

response = llm.get_chat_esponse(prompt_template=str(prompe_template))
print(response)
