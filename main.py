"""Main module used to process image"""
import get_faces as fc
import get_emotion as ge
import make_json
import cv2
import get_age_gender as ag

def process_image(image_path):
    """Processes the provided image getting faces and information and returns them as array"""
    face_array = fc.main(image_path)
    emotion_array = ge.main(face_array)
    age_array, gender_array = ag.get_age_gender_from_image_array(face_array)
    return make_json.main(face_array, emotion_array, age_array, gender_array)

def process_stream():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        output, faces = fc.draw_faces(buffer)
        emotion_array = ge.get_emotion_from_camera_stream(output, faces)
        age_array, gender_array = ag.get_age_gender_from_camera_stream(output, faces)
        write_to_image(output, faces, emotion_array, age_array, gender_array)
        ret, final = cv2.imencode('.jpg', output)
        frame = final.tobytes()
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'

def write_to_image(image, faces, emotion_array, age_array, gender_array):
    index = 0
    for (x, y) in faces:
        text = gender_array[index] + age_array[index] + emotion_array[index]
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, cv2.BGR_COMMON['green'], 1.3)
