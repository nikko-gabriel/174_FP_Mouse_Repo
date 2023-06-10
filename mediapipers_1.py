import cv2
import mediapipe as mp
import pyautogui

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
index_x, index_y = 0, 0
middle_x, middle_y = 0, 0
thumb_x, thumb_y = 0, 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:  # Index finger
                    cv2.circle(img=frame, center=(x, y), radius=25, color=(0, 255, 255))
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y

                elif id == 12:  # Middle finger
                    cv2.circle(img=frame, center=(x, y), radius=25, color=(0, 255, 255))
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y

                elif id == 4:  # Thumb
                    cv2.circle(img=frame, center=(x, y), radius=25, color=(0, 255, 255))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

            # Check distance between thumb and index finger for mouse movement
            if abs(index_y - thumb_y) < 30:
                pyautogui.moveTo(index_x, index_y)

            # Check distance between middle finger and index finger for left mouse click
            if abs(index_y - middle_y) < 30:
                pyautogui.click()
                pyautogui.sleep(1)

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
