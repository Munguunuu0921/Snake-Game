import math
import random
import cv2
import cvzone
import time
import numpy as np
from cvzone.HandTrackingModule import HandDetector
class SnakeGameClass:
    def __init__(self, pathFood):
        # Initialize variables for the snake game
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0  # previous head point
        # Load the initial food image with error handling
        self.foodEaten = 0
        self.foodImages = {
            "Food.png": 1,
            "Cake.png": 2,
            "Pizza.png": 3
        }  # Add more food types and points as needed
        self.currentFoodIndex = 0
        self.loadFoodImage()
        self.score = 0
        self.gameOver = False
        # Initialize foodPoint attribute
        self.foodPoint = 0, 0

    def loadFoodImage(self):
        # Load the current food image with error handling
        currentFoodImage = list(self.foodImages.keys())[self.currentFoodIndex]
        self.imgFood = cv2.imread(currentFoodImage, cv2.IMREAD_UNCHANGED)
        if self.imgFood is None:
            raise FileNotFoundError(f"Could not load the image: {currentFoodImage}")

        # Check if the loaded image has an alpha channel
        if self.imgFood.shape[2] == 4:
            self.hasAlphaChannel = True
        else:
            self.hasAlphaChannel = False

        self.hFood, self.wFood, _ = self.imgFood.shape

    def randomFoodLocation(self):
        # Set a random location for the food
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):
        if self.gameOver:
            # Display game over message
            # cvzone.putTextRect(imgMain, "Togloom duuslaa", [300, 400],
            #                    scale=7, thickness=5, offset=20,color=(251, 240, 178))
            # cvzone.putTextRect(imgMain, f'Onoo: {self.score}', [300, 550],
            #                    scale=7, thickness=5, offset=20,color=(251, 240, 178))
            cv2.putText(imgMain, "Togloom duuslaa", (300, 400),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 128, 128), 3)
            cv2.putText(imgMain, f'Onoo: {self.score}', (300, 550),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 128, 128), 3)
        else:
            px, py = self.previousHead
            cx, cy = currentHead

            # Update snake position and length
            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # Length Reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break

            # Check if snake ate the Food
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                    ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += self.foodImages[list(self.foodImages.keys())[self.currentFoodIndex]]
                self.foodEaten += 1

                # Check if snake has eaten 5 food items
                if self.foodEaten % 5 == 0:
                    self.currentFoodIndex = (self.currentFoodIndex + 1) % len(self.foodImages)
                    self.loadFoodImage()

                print(self.score)

            # Draw Snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, self.points[i - 1], self.points[i], (238, 147, 34), 20)
                cv2.circle(imgMain, self.points[-1], 20, (255, 187, 92), cv2.FILLED)

            # Draw Food
            if self.hasAlphaChannel:
                imgMain = cvzone.overlayPNG(imgMain, self.imgFood,
                                            (rx - self.wFood // 2, ry - self.hFood // 2))
            else:
                try:
                    imgMain[ry - self.hFood // 2:ry + self.hFood // 2,
                            rx - self.wFood // 2:rx + self.wFood // 2] = self.imgFood[:, :, :3]
                except ValueError:
                    print(f"Error: Could not broadcast input array from shape {self.imgFood.shape} into shape {imgMain.shape}")

            # cvzone.putTextRect(imgMain, f'Onoo: {self.score}', [50, 80],
            #                    scale=3, thickness=3, offset=10, color=(251, 240, 178))
            cv2.putText(imgMain, f'Onoo: {self.score}', (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 128, 128), 3)

            # Check for Collision
            pts = np.array(self.points[:-2], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(imgMain, [pts], False, (255, 181, 52), 3)
            minDist = cv2.pointPolygonTest(pts, (cx, cy), True)
            if -1 <= minDist <= 1:
                print("Hit")
                self.gameOver = True
                self.points = []  # all points of the snake
                self.lengths = []  # distance between each point
                self.currentLength = 0  # total length of the snake
                self.allowedLength = 150  # total allowed Length
                self.previousHead = 0, 0  # previous head point
                self.randomFoodLocation()

        return imgMain

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.8, maxHands=1)
    game = SnakeGameClass("Food.png")  # Use the first food image initially

    while True:
        success, img = cap.read()
        if not success:
            print("Error: Failed to capture frame.")
            time.sleep(1)
            continue

        img = cv2.flip(img, 1)

        if img is None:
            print("Error: Empty frame. Skipping.")
            time.sleep(1)
            continue

        hands, img = detector.findHands(img, flipType=False)
        if hands:
            lmList = hands[0]['lmList']
            pointIndex = lmList[8][0:2]
            img = game.update(img, pointIndex)

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == ord('r'):
            game.gameOver = False

if __name__ == "__main__":
    main()
main()