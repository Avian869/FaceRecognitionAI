"""Main module used to process image or camera stream"""
import get_faces as fc
import get_emotion
import make_json

def process_image(image_path):
    """Processes the provided image getting faces and information and returns them as array"""
    face_array = fc.main(image_path)
    emotion_array = get_emotion.main(face_array)
    return make_json.main(face_array, emotion_array)
