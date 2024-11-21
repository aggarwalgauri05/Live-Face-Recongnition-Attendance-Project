import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import streamlit as st

# Set up the image path and load images
path = 'imagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to find encodings for known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Generate encodings for known faces
encodeListKnown = findEncodings(images)

# Function to mark attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

# Set up webcam capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

# Streamlit UI setup
st.title("Attendance System")

# Create a placeholder for the video feed
frame_placeholder = st.empty()

# Button to stop video capture (only created once outside the loop)
stop_button = st.button("Stop Video", key="stop_video_button")

# Loop for continuous video feed
while not stop_button:
    success, img = cap.read()
    if not success:
        st.warning("Failed to grab frame.")
        break
    
    # Resize and convert the image to RGB
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find faces and their encodings in the current frame
    facesCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)

    # Process each face found
    for encodeFace, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    # Convert image to RGB for Streamlit and update the feed
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(imgRGB, channels="RGB", use_container_width=True)

    # Check if the stop button was pressed (the loop breaks if True)
    if stop_button:
        break

# Release the webcam when done
cap.release()
