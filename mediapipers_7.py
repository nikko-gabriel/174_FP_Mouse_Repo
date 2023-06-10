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
pinky_x, pinky_y = 0, 0

left_click_overlay = False
right_click_overlay = False
left_clicking_overlay = False
left_button_held = False

countdown_start = False
countdown_timer = 0
countdown_duration = 20

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

            wrist_landmark = landmarks[0]
            wrist_x = int(wrist_landmark.x * frame_width)
            wrist_y = int(wrist_landmark.y * frame_height)

            hand_size = int(abs(index_x - wrist_x) + abs(index_y - wrist_y))
            circle_radius = max(int(hand_size * 0.08), 25)
            click_collision_radius = max(int(hand_size * 0.08), 25)

            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=circle_radius, color=(0, 255, 255))
                    index_x = screen_width / frame_width * x
                    index_y = screen_height / frame_height * y

                elif id == 12:
                    cv2.circle(img=frame, center=(x, y), radius=circle_radius, color=(0, 255, 255))
                    middle_x = screen_width / frame_width * x
                    middle_y = screen_height / frame_height * y

                elif id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=circle_radius, color=(0, 255, 255))
                    thumb_x = screen_width / frame_width * x
                    thumb_y = screen_height / frame_height * y

                elif id == 20:
                    cv2.circle(img=frame, center=(x, y), radius=circle_radius, color=(0, 255, 0))
                    pinky_x = screen_width / frame_width * x
                    pinky_y = screen_height / frame_height * y

            if abs(index_x - thumb_x) < click_collision_radius and abs(index_y - thumb_y) < click_collision_radius:
                if not left_button_held:
                    pyautogui.mouseDown()
                    left_button_held = True
                    left_clicking_overlay = True

            else:
                if left_button_held:
                    pyautogui.mouseUp()
                    left_button_held = False

            if left_button_held:
                left_clicking_overlay = True

            if abs(middle_x - thumb_x) < click_collision_radius and abs(middle_y - thumb_y) < click_collision_radius:
                pyautogui.rightClick()
                right_click_overlay = True

            pyautogui.moveTo(pinky_x, pinky_y)

            if (
                abs(index_x - middle_x) < click_collision_radius and
                abs(middle_x - pinky_x) < click_collision_radius and
                thumb_y > index_y
            ):
                countdown_start = True
            else:
                countdown_start = False
                countdown_timer = 0

    if countdown_start:
        if countdown_timer < countdown_duration:
            countdown_text = "Closing in: {}".format(countdown_duration - countdown_timer)
            cv2.putText(frame, countdown_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            countdown_timer += 1
        else:
            break

    if left_click_overlay or left_clicking_overlay:
        cv2.putText(frame, 'Left Clicking', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        left_click_overlay = False
        left_clicking_overlay = False

    if right_click_overlay:
        cv2.putText(frame, 'Right Click', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        right_click_overlay = False

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
