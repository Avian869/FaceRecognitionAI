import numpy as np
import tensorflow as tf
import cv2
import numpy
import pathlib

folder_path = "static/images/"

interpreter = tf.lite.Interpreter(model_path="model.tflite")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

def get_age_gender_from_image_array(path_array):
    """This method gets all ages and genders from faces"""
    age_array = []
    gender_array = []
    for imgpath in path_array:
        face = cv2.imread('static\\{0}'.format(imgpath))
        new_img = cv2.resize(face, (96, 96)).astype('float32')/255
        interpreter.set_tensor(input_details[0]['index'], [new_img])
        interpreter.invoke()
        gender = 'female'
        if interpreter.get_tensor(output_details[1]['index']) < 0.95:
            gender = 'male'
        age_array.append(interpreter.get_tensor(output_details[0]['index']))
        gender_array.append(gender)
    return age_array, gender_array

def get_age_gender_from_camera_stream(image, faces):
    """This method gets all emotions from faces"""
    age_array = []
    gender_array = []
    for (x, y, w, h) in faces:
        img_crop = image[y:y+h+1, x:x+w+1]
        new_img = cv2.resize(img_crop, (96, 96)).astype('float32')/255
        interpreter.set_tensor(input_details[0]['index'], [new_img])
        interpreter.invoke()
        gender = 'female'
        if interpreter.get_tensor(output_details[1]['index']) < 0.95:
            gender = 'male'
        age_array.append(interpreter.get_tensor(output_details[0]['index']))
        gender_array.append(gender)
    return age_array, gender_array
