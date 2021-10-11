import mediapipe as mp
import cv2 
import numpy as np 
import time 
import os
import HandTrackingModule as htm 
import imutils
######################
brushThickness = 15
eraserThickness = 50
######################

folderpath="/home/shubharthak/Desktop/shubhi_handmodule/hand_detector_shubh/virtual_painter/headers"
myList = os.listdir(folderpath)
myList.sort()
print(myList)
overlaylist= []

for impath in myList:
    image = cv2.imread(f'{folderpath}/{impath}')
    overlaylist.append(image)
    
# print(len(overlaylist))
header = overlaylist[0]
drawColor = (255, 0, 255)

w,h = 1280, 720
dim = (w, h)
cap = cv2.VideoCapture(0)
detector = htm.handDetector(detectionConfidence=0.85)
xp, yp = 0, 0
imgCanvas = np.zeros((720,1280,3), np.uint8)
print(imgCanvas.shape)
while True:
    #1.import image
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    h , w, c = img.shape
    # print(h, w, c)
    #2. Find Hand Landmarks
    img = detector.findHands(img)
    lm = detector.findPosition(img, draw = False)
    if len(lm)!= 0:
        #tip of index and middle finger
        x1, y1 = lm[8][1:]
        x2, y2 = lm[12][1:]

        #3. Check which fingers are up
        fingers = detector.fingersUp()
        print(fingers)
        #4. If selection mode - Two finger are up 
        if fingers[1] and fingers[2]:
            xp, yp  = 0, 0
            print("Selection Mode")
            #checking for click 
            if y1 < 125:
                if 300 < x1 < 450:
                    drawColor = (255, 0, 255)
                    header = overlaylist[0]
                elif 550 < x1 < 750:
                    header = overlaylist[1]
                    drawColor = (255, 0, 0)
                elif  800 < x1 < 950:
                    header = overlaylist[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlaylist[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25), drawColor, cv2.FILLED)

        #5. If Drawing mode - Index finger is up 
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")
            #drawing 
            if xp == 0 and yp == 0:
                xp , yp = x1, y1
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp, yp = x1, y1
    # imgCanvas = cv2.resize(imgCanvas, dim,interpolation=cv2.INTER_AREA )
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = cv2.bitwise_and(img,imgInv) 
    img = cv2.bitwise_or(img, imgCanvas)
    #setting the header img 
    img[0:125, 0:1280] = header
    h , w, c = img.shape
    print(h, w, c)
    cv2.imshow("Virtual-Painter-by-Shubharthak", img)
    cv2.imshow("Canvas", imgCanvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()