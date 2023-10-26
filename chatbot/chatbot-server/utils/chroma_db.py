import chromadb


class DocumentQuery:
    def __init__(self) -> None:
        pass

    def initialize(self, collection_name="default", database="./database"):
        print("Initializing DocumentQueryBot...")
        self.chroma_client = chromadb.PersistentClient(path=database)
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print("Get collection from {}".format(database))
        except:
            self.collection = None
            print("Collection not exists.")

    def query(self, query_text, n_results=1):
        if not self.collection:
            print("No external database was loaded. Using raw input as LLM input...")
            return ""
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        matched_qas = results["metadatas"][0]
        print(matched_qas)
        res = [
            f"Q-{i}: {qa['question']}\nA-{i}: {qa['answer']}"
            for i, qa in enumerate(matched_qas)
        ]
        res = "\n\n".join(res)
        return res
