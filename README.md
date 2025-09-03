# Workflow Pipeline of System Architecture
<img width="1310" height="539" alt="Screenshot 2025-09-03 at 2 12 22 PM" src="https://github.com/user-attachments/assets/1fb8ccc3-bee6-41be-a1ac-1acd0f6078f2" />

## 1. Download Video from YouTube
We use [`yt-dlp`] to download videos in **144p** quality 

```bash
yt-dlp -f "bestvideo[height=144]+bestaudio/best[height=144]" "ytvideo.link"
```



## 2. Once the video is downloaded (in .webm format), use ffmpeg to convert it into an MP3 file.

```bash
ffmpeg -i "input_video_name.webm" output_audio_file.mp3
```
