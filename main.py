import getfaces as fc
import getEmotion
import makejson

def process_image(image_path):
    """Processes the provided image getting faces and information and returns them as array"""
    face_array = fc.main(image_path)
    emotion_array = getEmotion.main(face_array)
    return makejson.main(face_array, emotion_array)
