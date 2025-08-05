import random
from chat_assistant import Tools, ChatAssistant, ChatInterface

known_weather_data = {
    'berlin': 20.0
}

def get_weather(city: str) -> float:
    city = city.strip().lower()

    if city in known_weather_data:
        return known_weather_data[city]

    return round(random.uniform(-5, 35), 1)

def set_weather(city: str, temp: float) -> None:
    city = city.strip().lower()
    known_weather_data[city] = temp
    return 'OK'

get_weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Search the actual weather in a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to search the weather for."
                }
            },
            "required": ["city"],
            "additionalProperties": False
        }
    }
}

set_weather_tool = {
    "type": "function",
    "function": {
        "name": "set_weather",
        "description": "Set the weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to search the weather for."
                },
                "temp": {
                    "type": "string",
                    "description": "The temperature to set for the city."
                }
            },
            "required": ["city", "temp"],
            "additionalProperties": False
        }
    }
}

tools = Tools()
tools.add_tool(get_weather, get_weather_tool)
tools.add_tool(set_weather, set_weather_tool)
chat_assistant = ChatAssistant(
    tools=tools,
    developer_prompt="You are a helpful assistant that can answer questions about the weather.",
    chat_interface=ChatInterface()
)
chat_assistant.run()