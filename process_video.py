import whisper
import os
import subprocess

files = os.listdir("videos")
#print(files)

for file in files:
    #print(file)
    file_num = file.split(" [")[0].split(" #")[1]
    file_name = file.split(" ｜")[0]
    print(file_num, file_name)
    subprocess.run(["ffmpeg", "-i",f"videos/{file}", f"audios/{file_num}_{file_name}.mp3"])
 