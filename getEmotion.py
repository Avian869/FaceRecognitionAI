from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

def GetEmotion(face):
    classifier = load_model('EmotionDetectionModel.h5')
    class_labels=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
    gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
    roi_gray=cv2.resize(gray,(48,48),interpolation=cv2.INTER_AREA)
    roi=roi_gray.astype('float')/255.0
    roi=img_to_array(roi)
    roi=np.expand_dims(roi,axis=0)
    preds=classifier.predict(roi)[0]
    label=class_labels[preds.argmax()]
    return label

def main(faceArray):
    i = 0
    emotionArray = []
    f = open('Emotions.txt','w')
    for face in faceArray:
        emotionArray.append(GetEmotion(face))
        f.write(emotionArray[i] + '\n')
        i += 1
    f.close
    return emotionArray