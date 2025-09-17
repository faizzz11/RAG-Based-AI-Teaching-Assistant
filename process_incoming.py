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
# print(similarities)  # it will give similarity of question with all the chunks  close to 1 will be highest similar and close to 0 will be least similar

top_result = 5
max_indx = similarities.argsort()[::-1][0:top_result] # returns the indices that would sort an array. It does not return the sorted values — it returns the order of indices. so here if we do [::-1] it will reverse the order and [0:3] will give top 3 indices
# print(max_indx) # printing top 3 indices

new_df =  df.loc[max_indx]
# print(new_df[["title","number", "id" , "text"]])  # it will give top 3 similar chunk


prompt = f"""
You are acting as a friendly course guide for students following the Sigma Web Development Course.

You’ll be given subtitle chunks from the course videos. Each chunk includes:  
- Video title  
- Video number  
- Chunk ID  
- Start time in seconds  
- End time in seconds  
- Transcript text  

Your job:  
- Use the subtitle data to answer the student’s question clearly.  
- Mention **video number and title** where the answer can be found.  
- Convert all timestamps from seconds to **minutes:seconds format** (for example, 125 seconds → 2:05).  
- Provide a list or timeline of the relevant timestamps so the student can jump directly to those parts of the video.  
- Speak directly to the student (like a tutor, not a teacher).  
- If the question is off-topic or unrelated to the course, politely explain you can only answer questions about the course content.  
- If you don’t have enough info to answer, reply with: “I’m not sure about that.”  

Here are the subtitle chunks you can use:  
{new_df[["title", "number", "id", "start", "end", "text"]].to_json(orient="records", lines=False)}

--------------------------------------------------------------------------------------------------------
Student’s Question: "{incoming_query}"
"""


with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)



# for index, item in new_df.iterrows():
#     print(index)
#     print(f"Title: {item['title']}")
#     print(f"Video Number: {item['number']}")
#     print(f"Chunk ID: {item['id']}")
#     print(f"Text: {item['text']}")
#     print(f"start: {item['start']/60}")
#     print(f"end: {item['end']/60}")
#     print("\n---\n")