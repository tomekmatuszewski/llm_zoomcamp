
import json
from typing import Any

import html2text
from ollama import chat
from ollama import ChatResponse
import markdown


class Tools:
    def __init__(self):
        self.tools = {}
        self.functions = {}

    def add_tool(self, function, description):
        self.tools[function.__name__] = description
        self.functions[function.__name__] = function
    
    def get_tools(self):
        return list(self.tools.values())

    def function_call(self, tool_call_response: dict[str, Any]):
        function_name = tool_call_response["name"]
        arguments = tool_call_response["arguments"]

        f = self.functions[function_name]
        result = f(**arguments)

        return {
            "role": "tool",
            "content": json.dumps(result, indent=2),
        }


def shorten(text, max_length=50):
    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


class ChatInterface:
    def input(self):
        question = input("You:")
        return question
    
    def display(self, message):
        print(message)

    def display_function_call(self, entry, result):
        call_html = f"""
            <details>
            <summary>Function call: <tt>{entry["name"]}({shorten(entry["arguments"])})</tt></summary>
            <div>
                <b>Call</b>
                <pre>{entry}</pre>
            </div>
            <div>
                <b>Output</b>
                <pre>{result['content']}</pre>
            </div>
            
            </details>
        """
        print(html2text.html2text(call_html))

    def display_response(self, message):
        response_html = markdown.markdown(message["content"])
        html = f"""
            <div>
                <div><b>Assistant:</b></div>
                <div>{response_html}</div>
            </div>
        """
        print(html2text.html2text(html))



class ChatAssistant:
    def __init__(self, tools: Tools, developer_prompt, chat_interface):
        self.tools = tools
        self.developer_prompt = developer_prompt
        self.chat_interface = chat_interface
    
    def llama(self, chat_messages) -> ChatResponse:
        return chat(
            model='llama3.1:8b',
            messages=chat_messages,
            tools=self.tools.get_tools(),
        )


    def run(self):
        chat_messages = [
            {"role": "developer", "content": self.developer_prompt},
        ]

        # Chat loop
        while True:
            question = self.chat_interface.input()
            if question.strip().lower() == 'stop':  
                self.chat_interface.display("Chat ended.")
                break

            message = {"role": "user", "content": question}
            chat_messages.append(message)

            while True:  # inner request loop
                response = self.llama(chat_messages)
                dumped_response = response.model_dump()
                chat_messages.append(dumped_response["message"])
                message = dumped_response["message"]
                print('Response:', dumped_response)

                if message["tool_calls"]:
                    for entry in message["tool_calls"]:
                        result = self.tools.function_call(entry["function"])
                        chat_messages.append(result)
                        self.chat_interface.display_function_call(entry["function"], result)
                else:
                    self.chat_interface.display_response(message)
                    break


