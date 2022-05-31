from __future__ import print_function
import cv2 as cv

face_cascade_name = 'haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'haarcascade_eye_tree_eyeglasses.xml'
face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()
#-- 1. Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!)Error loading eyes cascade')
    exit(0)
image_path = 'aim_face.jpg'
#-- 2. Read the image
img = cv.imread(image_path)

#-- 3. Run detector
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = cv.equalizeHist(img_gray)
#-- Detect faces
faces = face_cascade.detectMultiScale(img_gray)
for (x,y,w,h) in faces:
    center = (x + w//2, y + h//2)
    img = cv.ellipse(img, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
    faceROI = img_gray[y:y+h,x:x+w]
    #-- In each face, detect eyes
    eyes = eyes_cascade.detectMultiScale(faceROI)
    for (x2,y2,w2,h2) in eyes:
        eye_center = (x + x2 + w2//2, y + y2 + h2//2)
        radius = int(round((w2 + h2)*0.25))
        img = cv.circle(img, eye_center, radius, (255, 0, 0 ), 4)
cv.imshow('Capture - Face detection', img)

cv.waitKey()