import cv2
import enchant
import numpy as np
import pytesseract
import re
from autocorrect import Speller
from PIL import Image



#add libraries

from gtts import gTTS
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
import os

# Initialize dictionaries and spell checker
d = enchant.Dict("en_US")
spell = Speller(lang='en')

# Function to crop image by removing a number of pixels
def shrinkByPixels(im, pixels):
    h, w = im.shape[:2]
    return im[pixels:h-pixels, pixels:w-pixels]

# Function to adjust the gamma in an image by some factor
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Comparison function for sorting contours
def get_contour_precedence(contour, cols):
    tolerance_factor = 200
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]

# Function to find all speech bubbles in the given comic page and return a list of their contours
def findSpeechBubbles(image):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(imageGray, 235, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourMap = filterContoursBySize(contours)
    contourMap = filterContainingContours(contourMap, hierarchy)
    finalContourList = list(contourMap.values())
    finalContourList.sort(key=lambda x: get_contour_precedence(x, binary.shape[1]))
    return finalContourList

def filterContoursBySize(contours):
    contourMap = {}
    for i, contour in enumerate(contours):
        if 4000 < cv2.contourArea(contour) < 120000:
            epsilon = 0.0025 * cv2.arcLength(contour, True)
            approximatedContour = cv2.approxPolyDP(contour, epsilon, True)
            contourMap[i] = approximatedContour
    return contourMap

def filterContainingContours(contourMap, hierarchy):
    for i in list(contourMap.keys()):
        currentIndex = i
        while hierarchy[0][currentIndex][3] > 0:
            parentIndex = hierarchy[0][currentIndex][3]
            if parentIndex in contourMap:
                contourMap.pop(parentIndex)
            currentIndex = parentIndex
    return contourMap

def cropSpeechBubbles(image, contours, padding=0):
    croppedImageList = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        croppedImage = image[y-padding:y+h+padding, x-padding:x+w+padding]
        croppedImageList.append(croppedImage)
    return croppedImageList

def processScript(script):
    if "COMICS.COM" in script:
        return ''
    script = script.replace('|', 'I').replace('\n', ' ')
    script = re.sub(r'\s+', ' ', script)
    script = ''.join([char for char in script if char in ' -QWERTYUIOPASDFGHJKLZXCVBNM,.?!""\'’1234567890'])
    script = re.sub(r"(?<!-)- ", "", script)
    words = script.split()
    for i in range(len(words)):
        if not d.check(words[i]):
            alphaWord = ''.join([j for j in words[i] if j.isalpha()])
            if alphaWord and not d.check(alphaWord):
                words[i] = spell(words[i].lower()).upper()
        if len(words[i]) == 1 and words[i] not in ['I', 'A']:
            words[i] = ''
    script = ' '.join(words)
    if len(script) == 2 and script not in ["NO", "OK"]:
        return ''
    return script

def tesseract(image):
    script = pytesseract.image_to_string(image, lang='eng')
    return processScript(script)

def segmentPage(image, shouldShowImage=False):
    contours = findSpeechBubbles(image)
    croppedImageList = cropSpeechBubbles(image, contours)
    cv2.drawContours(image, contours, -1, (0, 0, 0), 2)
    if shouldShowImage:
        cv2.imshow('Speech Bubble Identification', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return croppedImageList


def parseComicSpeechBubbles(croppedImageList, shouldShowImage=False):
    scriptList = []
    for croppedImage in croppedImageList:
        # Resize image
        resizedImage = cv2.resize(croppedImage, (0, 0), fx=2, fy=2)
        
        # Preprocess image
        grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
        blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
        _, threshImage = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Optional image display for debugging
        if shouldShowImage:
            cv2.imshow('Processed Image', threshImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Perform OCR using Tesseract
        script = pytesseract.image_to_string(threshImage, lang='eng', config='--psm 6')
        
        # Retry with progressively smaller images
        count = 0
        while script.strip() == '' and count < 3:
            count += 1
            resizedImage = cv2.resize(resizedImage, (0, 0), fx=0.5, fy=0.5)
            grayImage = cv2.cvtColor(resizedImage, cv2.COLOR_BGR2GRAY)
            blurredImage = cv2.GaussianBlur(grayImage, (5, 5), 0)
            _, threshImage = cv2.threshold(blurredImage, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            script = pytesseract.image_to_string(threshImage, lang='eng', config='--psm 6')
        
        # Ensure script is valid and unique
        if script.strip() and script.strip() not in scriptList:
            scriptList.append(script.strip())
    
    return scriptList

def extract_text_from_comic(image_path, shouldShowImage=False):
    image = cv2.imread(image_path)
    croppedImageList = segmentPage(image, shouldShowImage)
    pageText = parseComicSpeechBubbles(croppedImageList, shouldShowImage)
    return pageText



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

def generate_video(text, image_path, output_video_path):
    audio_path = "temp_audio.mp3"
    text_to_speech(text, audio_path)
    create_video_with_image_and_audio(image_path, audio_path, output_video_path)
    os.remove(audio_path)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract text from comic images with speech bubbles.')
    parser.add_argument('image_path', type=str, help='Path to the comic image')
    parser.add_argument('--show', action='store_true', help='Show images during processing')

    args = parser.parse_args()
    # here is the extracted text
    text = extract_text_from_comic(args.image_path, args.show)
    print("image path is : ", args.image_path)
    generate_video(text[0],args.image_path, "output.mp4")
    for idx, line in enumerate(text, start=1):
        print(f"Speech Bubble {idx}: {line}")
