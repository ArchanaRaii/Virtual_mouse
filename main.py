import cv2
import numpy as np 
import mediapipe as mp
import autopy

wCam, hCam =640, 480 #width and height
frameR = 100 
smoothening = 7
plocX, plocY = 0,0
clocX, clocY = 0, 0
dragging = False #indication of whether mouse is moving or not

screen_width, screen_height = autopy.screen.size() #mouse size

cap = cv2.VideoCapture(0) 
cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence = 0.7)
mpDraw = mp.solutions.drawing_utils

def fingers_up(lmList):
    fingers =[]
    fingers.append(1 if lmList[4][0] > lmList[3][0] else 0) #thumb
    for id in range(8, 21, 4): #index to pinky
        fingers.append(1 if lmList[id][1] < lmList[id-2][1] else 0)
    return fingers

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * wCam), int(lm.y * hCam)
                lmList.append((cx,cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
        if lmList:
            x1, y1 = lmList[8] #index
            x2, y2 = lmList[12] #Middle
            x_thumb, y_thumb = lmList[4] #Thumb

            fingers = fingers_up(lmList)

            #move mouse
            if fingers[1] == 1 and fingers[2] == 0:
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, screen_width))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, screen_height))
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening
                autopy.mouse.move(screen_width - clocX, clocY)
                plocX, plocY = clocX, clocY

            #left click
            distance_click = np.hypot(x2 - x1, y2 - y1)
            if fingers[1] == 1 and fingers[2] == 1 and distance_click < 40:
                cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

            #right click
            distance_rclick = np.hypot(x_thumb - x1, y_thumb - y1)
            if fingers[0] == 1 and fingers[1] ==1 and distance_rclick < 40:
                cv2.circle(img, ((x_thumb + x1) //2, (y_thumb + y1) // 2), 15, (0, 0, 255), cv2.FILLED)
                autopy.mouse.click(button = autopy.mouse.Button.RIGHT)

            #dragging: Both fingers up and apart    
            if fingers[1] == 1 and fingers[2] == 1 and distance_click > 60:
                if not dragging:
                    autopy.mouse.toggle(down=True)
                    dragging = True

            #Drop: All fingers up (release)
            if all(f == 1 for f in fingers):
                if dragging:
                    autopy.mouse.toggle(down=False)
                    dragging = False

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
            
