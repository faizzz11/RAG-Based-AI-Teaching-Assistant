import requests
import os 
import json
import pandas as pd

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding


jsons = sorted([f for f in os.listdir("jsons") if f.endswith(".json")])
# print(jsons)
my_dicts = []
chunk_id = 0

for json_file in jsons: # it will load all the chunk of a particular video
    with open(f"jsons/{json_file}",encoding="utf-8") as f:
        content = json.load(f)
    print(f"creating embeddings for {json_file}")    
    embeddings = create_embedding([c["text"] for c in content["chunks"]])    
    for i, chunk in enumerate(content["chunks"]):
        chunk["chunk_id"] = chunk_id
        chunk["embedding"] = embeddings[i]
        chunk_id +=1
        my_dicts.append(chunk)

print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
print(df)
# a = create_embedding(["cat sat on a mat", " I am sitting on a mat"])    
# print(a)

df.to_csv("all_chunks_&_embeddings.csv")