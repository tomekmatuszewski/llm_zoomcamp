import json
from llm import Llm
import requests
from minsearch_engine import MiniSearchEngine
from minsearch.append import AppendableIndex


docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()
documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)


MODEL = 'llama3.1:8b'

prompt_template = """
You're a course teaching assistant.

You're given a QUESTION from a course student and that you need to answer with your own knowledge and provided CONTEXT.

The CONTEXT is build with the documents from our FAQ database.
SEARCH_QUERIES contains the queries that were used to retrieve the documents
from FAQ to and add them to the context.
PREVIOUS_ACTIONS contains the actions you already performed.

At the beginning the CONTEXT is empty.

You can perform the following actions:

- Search in the FAQ database to get more data for the CONTEXT
- Answer the question using the CONTEXT
- Answer the question using your own knowledge

For the SEARCH action, build search requests based on the CONTEXT and the QUESTION.
Carefully analyze the CONTEXT and generate the requests to deeply explore the topic. 
If context doesn't provide enough information, use your own knowledge to generate the search queries.
Do not answer that context is not enough, just give the best possible answer with own knowledge. Do not answer that you can provide information
using own knwoledge, just give the answer - use then template for OWN_KNOWLEDGE listed as third below.

Don't use search queries used at the previous iterations.

Don't repeat previously performed actions.

Don't perform more than {max_iterations} iterations for a given student question.
The current iteration number: {iteration_number}. If we exceed the allowed number 
of iterations, give the best possible answer with the provided information or if provided informaction are not enough own knowledge.


Output templates:

If you want to perform search, use this template:

{{
"action": "SEARCH",
"reasoning": "<add your reasoning here>",
"keywords": ["search query 1", "search query 2", ...]
}}

If you can answer the QUESTION using CONTEXT, use this template:

{{
"action": "ANSWER_CONTEXT",
"answer": "<your answer>",
"source": "CONTEXT"
}}

If you can't answer the QUESTION using CONTEXT, but you can answer it using your own knowledge, use this template:

{{
"action": "ANSWER",
"answer": "<your answer>",
"source": "OWN_KNOWLEDGE"
}}


Full answer should be only the JSON object from templates above, without any additional text around it.

<QUESTION>
{question}
</QUESTION>

<SEARCH_QUERIES>
{search_queries}
</SEARCH_QUERIES>

<CONTEXT> 
{context}
</CONTEXT>

<PREVIOUS_ACTIONS>
{previous_actions}
</PREVIOUS_ACTIONS>
""".strip()

engine = MiniSearchEngine(index=AppendableIndex(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"]
), documents=documents)

def build_context(search_results):
    context = ""

    for doc in search_results:
        context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
    
    return context.strip()

def dedup(seq: list[dict[str, str]]):
    seen = set()
    result = []
    for elem in seq:
        if elem["_id"] not in seen:
            result.append(elem)
            seen.add(elem["_id"])
    return result

question = 'what do I need to do to be successful at module 1?'
max_iterations = 3
iteration_number = 1
search_queries = []
search_results  = []
previous_actions = []

llm = Llm(model=MODEL)

for iteration in range(max_iterations):
    print(f'ITERATION #{iteration}...')
    context = build_context(search_results)

    prompt = prompt_template.format(
        question=question,
        context=context,
        search_queries="\n".join(search_queries),
        previous_actions='\n'.join([json.dumps(a) for a in previous_actions]),
        max_iterations=max_iterations - 1,
        iteration_number=iteration
    )
    raw_response = llm.get_chat_esponse(prompt)
    answer = json.loads(raw_response)
    print(f'LLM response: {answer}')
    print(json.dumps(answer, indent=2))
    previous_actions.append(answer)

    action = answer['action']
    if action != 'SEARCH':
        break

    for k in answer['keywords']:
        res = engine.search(
            query=question,
            boost_dict={"question": 3, "section": 0.5},
            filter_dict={"course": "data-engineering-zoomcamp"},
            num_result=5
        )
        search_results.extend(res)
    search_results = dedup(search_results)
