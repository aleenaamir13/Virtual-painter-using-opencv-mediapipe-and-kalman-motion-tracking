import cv2
import numpy as np
import os
import HandTrackingModule as htm

brushThickness = 25
eraserThickness = 100

folderPath = "Header"
myList = os.listdir(folderPath)

overlayList = []
for imPath in myList:
    image = cv2.imread(f"{folderPath}/{imPath}")
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.78, maxHands=1)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

# kalman filter
kalman = cv2.KalmanFilter(4, 2, 0, cv2.CV_32F)

kalman.measurementMatrix = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0]],
    dtype=np.float32
)

kalman.transitionMatrix = np.array(
    [[1, 0, 1, 0],
     [0, 1, 0, 1],
     [0, 0, 1, 0],
     [0, 0, 0, 1]],
    dtype=np.float32
)

kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1

initialized = False
newStroke = True

while True:

    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        if not initialized:
            kalman.statePre = np.array(
                [[float(x1)],
                 [float(y1)],
                 [0.0],
                 [0.0]],
                dtype=np.float32
            )
            kalman.statePost = kalman.statePre.copy()
            initialized = True

        # Selection mode
        if fingers[1] and fingers[2]:

            newStroke = True
            xp, yp = 0, 0

            if y1 < 125:

                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 0)

                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 255)

                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)

                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)

            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # drawing mode
        elif fingers[1] and not fingers[2]:

            measurement = np.array(
                [[float(x1)],
                 [float(y1)]],
                dtype=np.float32
            )

            # RESET KALMAN FOR NEW STROKE
            if newStroke:
                kalman.statePre = np.array(
                    [[float(x1)],
                     [float(y1)],
                     [0.0],
                     [0.0]],
                    dtype=np.float32
                )
                kalman.statePost = kalman.statePre.copy()
                newStroke = False

            kalman.correct(measurement)
            prediction = kalman.predict()

            xk = int(prediction[0, 0])
            yk = int(prediction[1, 0])

            cv2.circle(img, (xk, yk), 15, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = xk, yk

            distance = np.hypot(xk - xp, yk - yp)

            if distance > 3:

                thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness

                cv2.line(img, (xp, yp), (xk, yk), drawColor, thickness, cv2.LINE_AA)
                cv2.line(imgCanvas, (xp, yp), (xk, yk), drawColor, thickness, cv2.LINE_AA)

                xp, yp = xk, yk

        else:
            # NO DRAWING → RESET STROKE
            xp, yp = 0, 0
            newStroke = True

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)

    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)

    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:125, 0:1280] = header

    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
