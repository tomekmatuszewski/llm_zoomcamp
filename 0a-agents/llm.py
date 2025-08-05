from ollama import chat
from ollama import ChatResponse


class Llm:

    def __init__(self, model: str):
        self.model = model

    def get_chat_esponse(self, prompt_template: str) -> str:
        response: ChatResponse = chat(model=self.model, messages=[
            {
                'role': 'user',
                'content': prompt_template,
            },
        ])
        return response.message.content
    
    def get_response_with_tools(self, input: list, tools: list) -> ChatResponse:
        response: ChatResponse = chat(model=self.model, messages=input, tools=tools)
        return response