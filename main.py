import os
import numpy as np
import getfaces as fc
import getEmotion
import makejson

def process_image(image_path):
    """Processes the provided image getting faces and information and returns them as array"""
    face_array = fc.main(image_path)
    emotion_array = getEmotion.main(face_array)
    #result_array = np.column_stack((face_array, emotion_array))
    #resultArray = emotionArray
    #return resultArray
    return makejson.main(face_array, emotion_array)
    #return result_array.tolist()
