import os
import json
from tqdm import tqdm
import hashlib
import chromadb


def preprocess_data(input_folder):
    chroma_client = chromadb.PersistentClient(path="./database")
    collection_name = "default"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print("Get collection...")
    except:
        collection = chroma_client.create_collection(name=collection_name)
        print("Collection not exists, created:", collection_name)

    doc_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]

    doc_data = []

    total_lines = sum(1 for f in doc_files for _ in open(f))
    with tqdm(total=total_lines, desc="Loading data") as pbar:
        for f in doc_files:
            try:
                with open(f, "r") as fin:
                    for line in fin:
                        data = json.loads(line)
                        question = ""
                        answer = ""
                        for d in data["messages"]:
                            if d["role"] == "user":
                                question = d["content"]
                            elif d["role"] == "assistant":
                                answer = d["content"]
                        doc_id = hashlib.md5(question.encode()).hexdigest()
                        qa_content = "Question: {}\nAnswer: {}".format(question, answer)
                        doc_data.append((doc_id, question, answer, qa_content))
                        pbar.update(1)
            except:
                raise Exception("Check data AGAIN!")

    answer_dict = {data[0]: data[2] for data in doc_data}

    documents = [data[1] for data in doc_data]
    metadatas = [{"question": data[1], "answer": data[2]} for data in doc_data]
    ids = [str(data[0]) for data in doc_data]

    for i in tqdm(range(len(documents))):
        collection.upsert(
            documents=[documents[i]], metadatas=[metadatas[i]], ids=[ids[i]]
        )
    print("FINISHED.")


if __name__ == "__main__":
    input_folder = "./upload"
    preprocess_data(input_folder)
