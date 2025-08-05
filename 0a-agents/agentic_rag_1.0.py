import json
from llm import Llm
from minsearch.append import AppendableIndex
import requests

MODEL = 'llama3.1:8b'

class MiniSearchEngine:
    def __init__(self, index: AppendableIndex, documents: list[dict[str, str]]) -> None:
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


prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.
At the beginning the context is EMPTY.

<QUESTION>
{question}
</QUESTION>

<CONTEXT> 
{context}
</CONTEXT>

If CONTEXT == EMPTY, you can use our FAQ database.
In this case, use the following output template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>"
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If the context doesn't contain the answer (is as empty string), use your own knowledge to answer the question

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}
Respone should contain only json format 
""".strip()

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()
documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

engine = MiniSearchEngine(
    index=AppendableIndex(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
    ),
    documents=documents
)

def build_context(search_results):
    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    return context.strip()

def agentic_rag_v1(question):
    context = "EMPTY"
    prompt = prompt_template.format(question=question, context=context)
    llm = Llm(model=MODEL)
    answer_json = llm.get_chat_esponse(prompt)
    answer = json.loads(answer_json)
    print(answer)

    if answer['action'] == 'SEARCH':
        print('need to perform search...')
        search_results = engine.search(
            query=question,
            boost_dict={"question": 3, "section": 0.5},
            filter_dict={"course": "data-engineering-zoomcamp"},
            num_result=5
        )
        context = build_context(search_results)
        
        prompt = prompt_template.format(question=question, context=context)
        answer_json = llm.get_chat_esponse(prompt)
        answer = json.loads(answer_json)

    return answer

question = "what is the difference between a data lake and a data warehouse?"
answer = agentic_rag_v1(question)
print(answer)