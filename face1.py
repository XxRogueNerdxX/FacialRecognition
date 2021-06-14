import face_recognition
import cv2
import numpy as np

#import io
#import picamera



video_capture = cv2.VideoCapture(1)

img1 = face_recognition.load_image_file("1.jpeg")
img1_encoding = face_recognition.face_encodings(img1)[0]

img2 = face_recognition.load_image_file("2.jpg")
img2_encoding = face_recognition.face_encodings(img2)[0]

known_face_encodings = [
    img1_encoding,
    img2_encoding
]
known_face_names = [
    "Lena",
    "Biden"
]

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            print(known_face_names[best_match_index])
            #add i/o here
    process_this_frame = not process_this_frame
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
