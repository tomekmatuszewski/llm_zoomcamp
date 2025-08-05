from minsearch.append import AppendableIndex

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
            filter_dict=filter_dict,
            output_ids=True
        )
        return results