import os
import glob
from pathlib import Path
import cv2
import numpy

casc_path = "Cascades\\data\\haarcascade_frontalface_alt2.xml"
face_cascade = cv2.CascadeClassifier(casc_path)

def find_faces(image):
    """Finds all faces in a given image and returns an image array with all faces"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def draw_faces(image_par):
    """Saves all individual faces to temporary directory in root folder"""
    npimg = numpy.fromstring(image_par, numpy.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    faces = find_faces(image)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image, faces

def save_faces(faces, image):
    """Saves all individual faces to temporary directory in root folder"""
    i = 0
    face_array = []
    for (x, y, w, h) in faces:
        i += 1
        img_crop = image[y:y+h+1, x:x+w+1]
        filepath = 'images\\Img{0}.jpg'.format(i)
        cv2.imwrite('static\\{0}'.format(filepath), img_crop)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
        face_array.append(filepath)
    return face_array

def remove_results():
    """Removes all files from Results directory"""
    path = Path(__file__).parent / "Results"
    files = glob.glob(os.path.join(path, "*"))
    for file in files:
        try:
            os.remove(file)
        except OSError as exception:
            print("Error: %s : %s" % (file, exception.strerror))

def main(image_par):
    """Finds all faces in image and returns all locations of saved faces"""
    remove_results()
    filestr = image_par.read()
    npimg = numpy.fromstring(filestr, numpy.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    faces = find_faces(image)
    face_array = save_faces(faces, image)
    return face_array
