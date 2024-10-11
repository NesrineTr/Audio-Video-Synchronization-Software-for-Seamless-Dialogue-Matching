import tkinter as tk
from tkinter import filedialog
from gtts import gTTS
from PIL import Image
from moviepy.editor import ImageClip, AudioFileClip
import os

def text_to_speech(text, output_audio_path):
    tts = gTTS(text)
    tts.save(output_audio_path)

def create_video_with_image_and_audio(image_path, audio_path, output_video_path):

    image = Image.open(image_path)
    

    image_clip = ImageClip(image_path)
    

    audio_clip = AudioFileClip(audio_path)
    

    image_clip = image_clip.set_duration(audio_clip.duration)
    

    video_clip = image_clip.set_audio(audio_clip)
    

    video_clip.write_videofile(output_video_path, codec="libx264", fps=24)

def main(text, image_path, output_video_path):

    audio_path = "temp_audio.mp3"
    

    text_to_speech(text, audio_path)
    

    create_video_with_image_and_audio(image_path, audio_path, output_video_path)
    

    os.remove(audio_path)

def browse_image():
    filename = filedialog.askopenfilename()
    if filename:
        image_path_var.set(filename)

def create_video():
    text = text_entry.get("1.0", tk.END).strip()
    image_path = image_path_var.get()
    output_video_path = output_entry.get().strip()
    if not output_video_path.endswith(".mp4"):
        output_video_path += ".mp4"
    main(text, image_path, output_video_path)
    status_label.config(text=f"Video {output_video_path} created successfully!")


app = tk.Tk()
app.title("Text to Video Converter")

tk.Label(app, text="Enter Text:").pack()
text_entry = tk.Text(app, height=10, width=50)
text_entry.pack()

tk.Label(app, text="Select Image:").pack()
image_path_var = tk.StringVar()
image_entry = tk.Entry(app, textvariable=image_path_var, width=50)
image_entry.pack()
browse_button = tk.Button(app, text="Browse", command=browse_image)
browse_button.pack()

tk.Label(app, text="Output Video Name:").pack()
output_entry = tk.Entry(app, width=50)
output_entry.pack()

create_button = tk.Button(app, text="Create Video", command=create_video)
create_button.pack()

status_label = tk.Label(app, text="")
status_label.pack()

app.mainloop()
