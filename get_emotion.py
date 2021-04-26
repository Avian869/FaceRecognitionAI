from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

def GetEmotion(path_array):
    model = load_model('EmotionDetectionModel.h5')
    class_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    emotion_array = []
    for imgpath in path_array:
        face = cv2.imread('static\\{0}'.format(imgpath))
        gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        roi_gray=cv2.resize(gray,(48,48),interpolation=cv2.INTER_AREA)
        roi=roi_gray.astype('float')/255.0
        roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)
        preds=model.predict(roi)[0]
        emotion_array.append(class_labels[preds.argmax()])
    return emotion_array

def main(face_array):
    return GetEmotion(face_array)
    