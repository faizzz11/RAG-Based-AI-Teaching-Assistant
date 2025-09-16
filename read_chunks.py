import requests
import os 
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        #if(i==5):  #stoping for running example faster
        # break
    break    

# print(my_dicts)

df = pd.DataFrame.from_records(my_dicts)
#print(df)
# df.to_csv("all_chunks_&_embeddings.csv")
df.to_csv("testt.csv")

# a = create_embedding(["cat sat on a mat", " I am sitting on a mat"])    
# print(a)

incoming_query = input("Enter your query: ")
question_embedding = create_embedding([incoming_query])[0] # creating vector embedding of the user's question for futhur matching in future
# print(question_embedding)

# find similarities of question_embedding with other embeddings
# print(np.vstack(df["embedding"].values))  # converting 1d array into 2d arrays 
# print(np.vstack(df["embedding"].values).shape)

similarities = cosine_similarity(np.vstack(df["embedding"].values), [question_embedding]).flatten()
print(similarities)  # it will give similarity of question with all the chunks  close to 1 will be highest similar and close to 0 will be least similar

max_indx = similarities.argsort()[::-1][0:3] # returns the indices that would sort an array. It does not return the sorted values — it returns the order of indices. so here if we do [::-1] it will reverse the order and [0:3] will give top 3 indices
print(max_indx)

new_df =  df.loc[max_indx]
print(new_df[["title","number", "text"]])  # it will give top 3 similar chunk