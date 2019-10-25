import face_recognition
import cv2
import numpy as np
import smtplib
import os
import sys
from time import time
import datetime


video_capture = cv2.VideoCapture(0)
height=480 # set video widht
width=640 # set video height
if sys.version_info < (3, 0):
    video_capture.set(cv2.cv.CV_CAP_PROP_FPS, 30)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,  width)
    video_capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
else:
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
fps_frames = 0
fps_start = time()
framecount=0

vlad_image = face_recognition.load_image_file("Baza/vlad.jpg")
vlad_face_encoding = face_recognition.face_encodings(vlad_image)[0]

sveta_image = face_recognition.load_image_file("Baza/sveta1.jpg")
sveta_face_encoding = face_recognition.face_encodings(sveta_image)[0]

cornel_image = face_recognition.load_image_file("Baza/cornel.png")
cornel_face_encoding = face_recognition.face_encodings(cornel_image)[0]

iura_image = face_recognition.load_image_file("Baza/iura.jpg")
iura_face_encoding = face_recognition.face_encodings(iura_image)[0]

known_face_encodings = [
    vlad_face_encoding,
    sveta_face_encoding,
    cornel_face_encoding,
    iura_face_encoding
]
known_face_names = [
    "Vlad",
    "Sveta",
    "Cornel",
    "Iura"
]


face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


while True:
   
    ret, frame = video_capture.read()
    if ret == 0:
            break
    fps_frames += 1
    if (framecount!=0 and fps_frames >= framecount):
        break
    if (fps_frames % 30 == 29):
        fps = fps_frames / (time() - fps_start)
        fps_frames = 0
        fps_start = time()
    
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    
    rgb_small_frame = small_frame[:, :, ::-1]

    
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Necunoscut"

        
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

                
                face_names.append(name)

    process_this_frame = not process_this_frame
    
        
    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

       
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

   
    cv2.imshow('Video', frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
video_capture.release()
cv2.destroyAllWindows()

