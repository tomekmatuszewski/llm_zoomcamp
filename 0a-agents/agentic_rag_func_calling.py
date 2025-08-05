from minsearch_engine import MiniSearchEngine
from minsearch.append import AppendableIndex
import json
import requests
from llm import Llm
from typing import Any


MODEL = 'llama3.1:8b'

client = Llm(model=MODEL)

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()
documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)



index=AppendableIndex(
        text_fields=["question", "text", "section"],
        keyword_fields=["course"])

index.fit(documents)

def search(query):
    boost = {'question': 3.0, 'section': 0.5}

    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5,
        output_ids=True
    )

    return results

search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the FAQ database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}

def do_call(tool_call_response: dict[str, Any]):
    function_name = tool_call_response["name"]
    arguments = tool_call_response["arguments"]

    f = globals()[function_name]
    result = f(**arguments)

    return {
        "role": "tool",
        # "call_id": tool_call_response.call_id,
        "content": json.dumps(result, indent=2),
    }

developer_prompt = """
You're a course teaching assistant. 
You're given a question from a course student and your task is to answer it.

Use FAQ if your own knowledge is not sufficient to answer the question.
When using FAQ, perform deep topic exploration: make one request to FAQ,
and then based on the results, make more requests.

At the end of each response, ask the user a follow up question based on your answer.
""".strip()

chat_messages = [
    {"role": "developer", "content": developer_prompt},
]
tools = [search_tool]

while True: # main Q&A loop
    question = input() # How do I do my best for module 1?
    if question == 'stop':
        break

    message = {"role": "user", "content": question}
    chat_messages.append(message)

    while True: # request-response loop - query API till get a message
        response = client.get_response_with_tools(
            input=chat_messages,
            tools=tools
        )
        dumped_response= response.model_dump()
        chat_messages.append(dumped_response["message"])
        print('Response:', dumped_response["message"])
        message = dumped_response["message"]

        if message["tool_calls"]:
            for entry in message["tool_calls"]:
                print('function_call:', entry["function"]["name"])
                print()
                result = do_call(entry["function"])
                chat_messages.append(result)
                print('function_call_output:', result["content"])
        else:
            print(message["content"])
            print()
            break