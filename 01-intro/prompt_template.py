from dataclasses import dataclass

@dataclass
class PromptTemplate:
    query: str
    search_result: list[dict[str, str]]
    
    def __str__(self) -> str:
        return f"""
          Answer the question based on the aswers provided.
          If the question cannot be answered based on the context, say "I don't know".\n
          Question: {self.query}\n
          Answer: {",\n\n".join([f'section: {el["section"]}, question: {el["question"]}, text: {el["text"]}' for el in self.search_result])}
        """.strip()
