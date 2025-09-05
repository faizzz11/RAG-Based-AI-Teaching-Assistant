import whisper

model = whisper.load_model("large-v2")
result = model.transcribe(audio= "audios/9_Id & Classes in HTML.mp3", language="hi", task="translate")
print(result["text"])