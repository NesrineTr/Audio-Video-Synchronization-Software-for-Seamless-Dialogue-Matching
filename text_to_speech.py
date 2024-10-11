from gtts import gTTS
from PIL import Image
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
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

if __name__ == "__main__":
    text = "It is ok to not have the speech bubble"
    image_path = "/home/asus/Téléchargements/projet3/image1.png"
    output_video_path = "output_video.mp4"
    
    main(text, image_path, output_video_path)
