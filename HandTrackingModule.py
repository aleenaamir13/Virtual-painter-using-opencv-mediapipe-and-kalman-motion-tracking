import cv2
import time
import mediapipe as mp
import math

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

class handDetector:
    def __init__(self, maxHands=2, detectionCon=0.7, trackCon=0.5):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.tipIds = [4, 8, 12, 16, 20]
        self.running_mode = VisionRunningMode.VIDEO

        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
            running_mode=self.running_mode,

        )

        self.detector = HandLandmarker.create_from_options(self.options)
        self.results = None
        self.frame_id = 0

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=imgRGB)

        self.frame_id += 1
        timestamp = self.frame_id

        self.results = self.detector.detect_for_video(mp_image, timestamp)
        return img


    def findPosition(self, img, handNo=0, draw=True):
        CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # index
            (0, 9), (9, 10), (10, 11), (11, 12),  # middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
            (5, 9), (9, 13), (13, 17)  # palm links
        ]
        self.lmList = []

        if self.results and self.results.hand_landmarks:
            if handNo < len(self.results.hand_landmarks):
                hand = self.results.hand_landmarks[handNo]
                h, w, _ = img.shape

                for id in range(len(hand)):
                    lm = hand[id]
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lmList.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                for a, b in CONNECTIONS:
                    x1, y1 = self.lmList[a][1], self.lmList[a][2]
                    x2, y2 = self.lmList[b][1], self.lmList[b][2]
                    if draw:
                        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return self.lmList

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()

        if not success:
            continue

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()