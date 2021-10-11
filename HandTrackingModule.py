import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode = False, max_hands = 2, detectionConfidence = 0.5, trackConfidence = 0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        #creating the handmodule (baseline)
        self.mpHands = mp.solutions.hands
        #create the handmodule
        self.hands = self.mpHands.Hands(self.mode, self.max_hands,self.detectionConfidence, self.trackConfidence)
        #draw the hands 
        self.mpDraw = mp.solutions.drawing_utils
        self.tipID= [4, 8, 12, 16, 20]

    
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(self.results.multi_hand_landmarks)
        #for displaying the multiple hands on screen 
        if self.results.multi_hand_landmarks:
            for each_hand in self.results.multi_hand_landmarks:
                if draw :
                    self.mpDraw.draw_landmarks(img, each_hand, self.mpHands.HAND_CONNECTIONS)
               
        return img  
    def findPosition(self, img, handNo = 0, draw = True ):
        self.lm= []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, land_mark in enumerate(myHand.landmark):
                    # print(id, land_mark)
                        #parsing to pos to int
                        h,w,c = img.shape
                        cx, cy = int(land_mark.x * w), int(land_mark.y * h)
                        # print(id, cx, cy)
                        self.lm.append([id, cx, cy])
                        if draw:
                            #testing a circle
                            cv2.circle(img,(cx, cy), 3, (255, 0, 0), cv2.FILLED)
        return self.lm
    def fingersUp(self):
        fingers = []
        
        #Thumb
        if self.lm[self.tipID[0]][1] < self.lm[self.tipID[0] - 1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        #4 Fingers
        for id in range(1,5):
            if self.lm[self.tipID[id]][2] < self.lm[self.tipID[id] -2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lm[p1][1:]
        x2, y2 = self.lm[p2][1:]
        cx, cy = (x1 + x2) //2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2,y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), t, cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), t, cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), t, cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
            
            
            
    # current_time = time.time()
    # fps = 1 / (current_time - previous_time)
    # previous_time = current_time
    # #display fps
    # cv2.putText(img, str("FPS: " + str(int(fps))), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, #parameter: (image, text, pos, fontname, fontscale, color, thickness)
    # (255, 0, 255), 3 )
    # cv2.imshow("Image", img)
    # cv2.waitKey(1)

def main():
    cap = cv2.VideoCapture(0)
    #displaying fps
    previous_time = 0
    current_time = 0
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm = detector.findPosition(img)
        if len(lm)!=0:
            print(lm[4])
        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time
        #display fps
        cv2.putText(img, str("FPS: " + str(int(fps))), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, #parameter: (image, text, pos, fontname, fontscale, color, thickness)
        (255, 0, 255), 3 )
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()