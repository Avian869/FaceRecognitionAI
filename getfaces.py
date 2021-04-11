import cv2
import sys
import os

def FindFaces(image):
    cascPath = "Cascades\\data\\haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
    )
    #print ("Found {0} faces!".format(len(faces)))
    return faces

def GetFaces(faces, image):
    # Draw a rectangle around the faces
    i = 0
    faceArray = []
    for (x, y, w, h) in faces:
        i += 1
        img_crop = image[y:y+h+1, x:x+w+1]
        faceArray.append(img_crop)
    return faceArray

# def main(imagePath):
#     image = cv2.imread(imagePath)
#     faces = FindFaces(image)
#     faceArray = GetFaces(faces, image)
#     return faceArray

def main(imagePar):
    import numpy
    RemoveResults()
    filestr = imagePar.read()
    npimg = numpy.fromstring(filestr, numpy.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    faces = FindFaces(image)
    faceArray = SaveFaces(faces, image)
    return faceArray

def SaveFaces(faces, image):
    # Draw a rectangle around the faces
    i = 0
    faceArray = []
    for (x, y, w, h) in faces:
        i += 1
        img_crop = image[y:y+h+1, x:x+w+1]
        cv2.imwrite('Results\\Img{0}.jpg'.format(i), img_crop)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 0), 2)
        faceArray.append(img_crop)
    return faceArray

def RemoveResults():
    import glob
    from pathlib import Path
    path = Path(__file__).parent / "Results"
    files = glob.glob(os.path.join(path, "*"))
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

def testmain(imagePath):
    RemoveResults()
    image = cv2.imread(imagePath)
    faces = FindFaces(image)
    faceArray = SaveFaces(faces, image)
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
    return faceArray