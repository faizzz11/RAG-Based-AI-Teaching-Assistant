import pandas as pd
import requests
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

df = joblib.load("embeddings_df.joblib")    

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
print(new_df[["title","number", "id" , "text"]])  # it will give top 3 similar chunk