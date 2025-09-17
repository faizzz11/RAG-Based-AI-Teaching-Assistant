import pandas as pd
import requests
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import json

def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    embedding = r.json()["embeddings"]
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False,
        # "max_new_tokens": 512,
        # "temperature": 0.1,
        # "top_p": 0.75,
        # "stop": ["###"]
    })

    response = r.json()
    return response

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
You are a course assistant helping students learn from the Sigma Web Development Course.

You’ll receive subtitle chunks from the course videos. Each chunk contains:
- Video title
- Video number
- Chunk ID
- Start time in seconds
- End time in seconds
- Transcript text

Your job:
1. **Start every answer with a one-sentence context that directly answers the student’s question.**  
   Example:  
   “ID and Class attributes in HTML are taught in Video 9 called ‘Id & Classes in HTML’.”
2. After that sentence, **show the relevant information in a clean structured format**:
   - Mention **Video Number + Title** at the top.
   - List each relevant segment with timestamps converted from **seconds to minutes:seconds format** (for example:  
     - 111 seconds → 1:51  
     - 356 seconds → 5:56  
     - 654 seconds → 10:54).
   - Give a short one-line description of what is explained in that segment.
3. End with a **Tip** line summarizing where to start watching.
4. **Do NOT greet or chat casually** (no “Hi there” or “I’d be happy to…”).
5. **Do NOT ask the student any questions at the end.** Only give the response.
6. Use bullet points or a time-coded guide (like the example below):

Example output style:
ID and Class attributes in HTML are taught in Video 9 called "Id & Classes in HTML".
Video 9: "Id & Classes in HTML"

• 0:48 – 0:50 → Definition of ID attribute
• 0:49 – 0:51 → Definition of class attribute
• 1:18 – 1:21 → Usage of ID and classes in CSS

Tip: Watch from around 0:49 onwards for a clear explanation of both attributes.

7. **No small talk** – be factual and focused.
8. If the question is unrelated to the course, reply with: “I can only answer questions related to this course.”
9. If not enough info is available, reply: “I’m not sure about that.”

Here are the subtitle chunks you can use:
{new_df[["title", "number", "id", "start", "end", "text"]].to_json(orient="records", lines=False)}

--------------------------------------------------------------------------------------------------------
Student’s Question: "{incoming_query}"
"""



with open("prompt.txt", "w", encoding="utf-8") as f:
    f.write(prompt)

response = inference(prompt)
print(response)

# Extract just the text portion
response_text = response.get("response", "")

with open("response.txt", "w", encoding="utf-8") as f:
    f.write(response_text)

# for index, item in new_df.iterrows():
#     print(index)
#     print(f"Title: {item['title']}")
#     print(f"Video Number: {item['number']}")
#     print(f"Chunk ID: {item['id']}")
#     print(f"Text: {item['text']}")
#     print(f"start: {item['start']/60}")
#     print(f"end: {item['end']/60}")
#     print("\n---\n")