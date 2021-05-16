def main(face_array, emotion_array, age_array, gender_array):
    """Create json from array parameters"""
    json_response = []
    length = len(face_array)
    for i in range(length):
        json_response.append({'src': '..\\static\\{0}'.format(face_array[i]),
         'emotion': emotion_array[i],
         'age' : round(float(age_array[i]), 0),
         'gender' : gender_array[i]})
    return json_response
