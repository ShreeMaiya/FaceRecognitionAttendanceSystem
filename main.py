import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the folder containing images for attendance
path = 'Images'
images = []  # List to hold all the images
classNames = []  # List to hold the names of the people in the images
mylist = os.listdir(path)  # List all files in the specified path
print(mylist)

# Loop through each file in the directory
for cl in mylist:
    curImg = cv2.imread(f'{path}/{cl}')  # Read the image
    images.append(curImg)  # Add image to the images list
    classNames.append(os.path.splitext(cl)[0])  # Add the name (without extension) to classNames
print(classNames)

# Function to find encodings for all images
def findEncodings(images):
    encodeList = []  # List to hold encodings
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
        encode = face_recognition.face_encodings(img)[0]  # Find the encoding for the image
        encodeList.append(encode)  # Add encoding to the list
    return encodeList

# Function to mark attendance and check if already recorded
def markAccess(name):
    now = datetime.now()  # Get the current date and time
    dateString = now.strftime('%Y-%m-%d')  # Format the date as a string
    timeString = now.strftime('%H:%M:%S')  # Format the time as a string

    with open('Attendance.csv', 'r+') as f:  # Open the CSV file to read and write
        myDataList = f.readlines()  # Read all lines in the file
        namelist = []  # List to hold names of people who have marked attendance
        for line in myDataList:
            entry = line.strip().split('  ')  # Split each line into components
            if len(entry) > 1:
                recorded_date = entry[1]
                recorded_time = entry[2]
                namelist.append((entry[0], recorded_date, recorded_time))  # Add name, date, and time to namelist

        # Check if the name and date are already in the list
        if not any(name == entry[0] and dateString == entry[1] for entry in namelist):
            f.writelines(f'\n{name}  {dateString}  {timeString}')  # Write the new entry to the file

# Encode the known images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Start the webcam
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()  # Capture a frame from the webcam
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize the frame for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB

    # Find face locations and encodings in the current frame
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    # Variable to track if any face is detected
    face_detected = False

    # Loop through each face found in the frame
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # Compare with known encodings
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Compute the distances
        matchIndex = np.argmin(faceDis)  # Find the best match index

        # If a match is found and the distance is below the threshold, use the corresponding name
        if matches[matchIndex] and faceDis[matchIndex] < 0.5:
            name = classNames[matchIndex].upper()  # Get the corresponding name
            markAccess(name)  # Mark attendance for the recognized person
        else:
            name = "Unknown"  # Otherwise, label as unknown

        # Set face_detected to True if a face is found (whether recognized or not)
        face_detected = True

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back up the face locations
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a rectangle around the face
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # Draw a filled rectangle for the name
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Put the name text

    # Only show "Move Closer!!!" if no face is detected at all
    if not face_detected:
        cv2.putText(img, "Move Closer!!!", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Add instruction text to the webcam feed in green
    cv2.putText(img, "Press 'q' to Exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the video feed with the rectangles, names, and instructions
    cv2.imshow('Webcam', img)

    # Press 'q' to break the loop and stop the webcam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
