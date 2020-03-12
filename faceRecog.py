from __future__ import print_function
import cv2
import argparse
import os, os.path

# Function to resize input images while retaining aspect ratio
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# Function to detect faces from webcam input
def detectAndDisplay(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for num, (x,y,w,h) in enumerate(faces):
        frame = cv2.rectangle(frame, (x, y) ,(x+w, y+h), (255, 255, 255), 2) # frame with face detected        
        cropped = frame[y:y+h, x:x+w] # cropped faces
        recropped = ResizeWithAspectRatio(cropped, width=250, height=250) # resize cropped faces
        savePath = "D:\\Cropped Webcam Faces"
        cv2.imwrite(os.path.join(savePath , 'image_' + str(num) + '.jpg'), recropped)
        
    cv2.imshow('Capture - Face detection', frame)

parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='Camera divide number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade

face_cascade = cv2.CascadeClassifier()

# Loading the Haar Cascade to detect faces
if not face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml"): #cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)

camera_device = args.camera
# Read the video stream
cap = cv2.VideoCapture(camera_device)
if not cap.isOpened:
    print('--(!)Error opening video capture')
    exit(0)
while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break
    
    num = 1

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        frame = cv2.rectangle(frame, (x, y) ,(x+w, y+h), (255, 255, 255), 2) # frame with face detected        
        cropped = frame[y:y+h, x:x+w] # cropped faces
        recropped = ResizeWithAspectRatio(cropped, width=250, height=250) # resize cropped faces
        savePath = "D:\\Cropped Webcam Faces"
        cv2.imwrite(os.path.join(savePath , 'image_' + str(num) + '.jpg'), recropped)
        num += 1
        
    cv2.imshow('Capture - Face detection', frame)

    #detectAndDisplay(frame)
    if cv2.waitKey(10) == 27:
        break
