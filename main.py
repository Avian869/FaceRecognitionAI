
def ProcessImage(imagePath):
    import getfaces as fc
    import getEmotion
    import numpy as np
    faceArray = fc.main(imagePath)
    emotionArray = getEmotion.main(faceArray)
    #resultArray = np.column_stack((faceArray, emotionArray))
    #return resultArray.tolist()
    resultArray = emotionArray
    return resultArray


def TestProcessImage(imagePath):
    import getfaces as fc
    import getEmotion
    faceArray = fc.testmain(imagePath)
    getEmotion.main(faceArray)