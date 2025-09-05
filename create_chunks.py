import whisper
import os
import json

model = whisper.load_model("medium") #dowloading the whisper model which generates english text from different languages audios too
audios = os.listdir("audios")

for audio in audios:
    #print(audio)
    if("_" in audio):
        number = audio.split("_")[0]

        title = audio.split("_")[1][:-4]
        print(number,title)
        result = model.transcribe(audio= f"audios/{audio}", language="hi", task="translate", word_timestamps=False)
        chunks = []
        for segment in result['segments']:
            chunks.append({
            "number":number,
            "title":title,   
            "id":segment['id'],
            "start":segment['start'],
            "end":segment['end'],
           "text":segment['text']
            })

        chunks_with_metadata = {"chunks":chunks, "text": result['text']}

        print(chunks)
        with open(f"jsons/{number} {title}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f)
        
        
