import cv2  # Import the OpenCV library for image processing
import mediapipe as mp  # Import the Mediapipe library for hand tracking
import pyautogui  # Import the PyAutoGUI library for controlling the mouse

# set pyautogui.FAILSAFE as false
pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)  # Open the default camera
hand_detector = mp.solutions.hands.Hands()  # Create a hand tracking object
drawing_utils = mp.solutions.drawing_utils  # Utility functions for drawing landmarks
screen_width, screen_height = pyautogui.size()  # Get the screen size
index_x, index_y = 0, 0  # Initialize variables for storing index finger position
middle_x, middle_y = 0, 0  # Initialize variables for storing middle finger position
thumb_x, thumb_y = 0, 0  # Initialize variables for storing thumb position
pinky_x, pinky_y = 0, 0  # Initialize variables for storing pinky finger position
middle_base_x, middle_base_y = 0, 0 

left_click_overlay = False  # Flag for indicating left click overlay
right_click_overlay = False  # Flag for indicating right click overlay
left_clicking_overlay = False  # Flag for indicating left click hold overlay
left_button_held = False  # Flag for indicating left mouse button hold

countdown_start = False  # Flag for indicating countdown start
countdown_timer = 0  # Countdown timer variable
countdown_duration = 20  # Countdown duration in seconds

while True:
    _, frame = cap.read()  # Read a frame from the camera
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_height, frame_width, _ = frame.shape  # Get the frame dimensions
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB format
    output = hand_detector.process(rgb_frame)  # Process the frame to detect hands
    hands = output.multi_hand_landmarks  # Get the detected hands

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)  # Draw landmarks on the frame
            landmarks = hand.landmark  # Get the landmarks of the hand

            wrist_landmark = landmarks[0]  # Get the wrist landmark
            wrist_x = int(wrist_landmark.x * frame_width)  # Calculate the wrist x-coordinate
            wrist_y = int(wrist_landmark.y * frame_height)  # Calculate the wrist y-coordinate

            constant = 1000
            hand_size = max(constant - int(abs(middle_base_x - wrist_x) + abs(middle_base_y - wrist_y)), 20)  # Calculate hand size
            circle_radius = max(int(hand_size * 0.09), 20)  # Calculate circle radius for visualization
            click_collision_radius = max(int(hand_size * 0.12), 25)  # Calculate click collision radius

            print(hand_size)
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)  # Calculate x-coordinate of the landmark
                y = int(landmark.y * frame_height)  # Calculate y-coordinate of the landmark

                if id == 8:
                    cv2.circle(img=frame, center=(x, y), radius=circle_radius, color=(0, 255, 255))  # Draw circle for index finger
                    index_x = screen_width / frame_width * x  # Convert index finger x-coordinate to screen coordinates
                    index_y = screen_height / frame_height * y  # Convert index finger y-coordinate to screen coordinates

                elif id == 12:
                    cv2.circle(img=frame, center=(x, y), radius=circle_radius, color=(0, 255, 255))  # Draw circle for middle finger
                    middle_x = screen_width / frame_width * x  # Convert middle finger x-coordinate to screen coordinates
                    middle_y = screen_height / frame_height * y  # Convert middle finger y-coordinate to screen coordinates

                elif id == 4:
                    cv2.circle(img=frame, center=(x, y), radius=circle_radius, color=(0, 255, 255))  # Draw circle for thumb
                    thumb_x = screen_width / frame_width * x  # Convert thumb x-coordinate to screen coordinates
                    thumb_y = screen_height / frame_height * y  # Convert thumb y-coordinate to screen coordinates

                elif id == 20:
                    cv2.circle(img=frame, center=(x, y), radius=circle_radius, color=(0, 255, 0))  # Draw circle for pinky finger
                    pinky_x = screen_width / frame_width * x  # Convert pinky finger x-coordinate to screen coordinates
                    pinky_y = screen_height / frame_height * y  # Convert pinky finger y-coordinate to screen coordinates

                elif id == 9:
                    middle_base_x = screen_width / frame_width * x  # Convert pinky finger x-coordinate to screen coordinates
                    middle_base_y = screen_height / frame_height * y  # Convert pinky finger y-coordinate to screen coordinates

            if abs(index_x - thumb_x) < click_collision_radius and abs(index_y - thumb_y) < click_collision_radius:
                if not left_button_held:
                    pyautogui.mouseDown()  # Hold down the left mouse button
                    left_button_held = True
                    left_clicking_overlay = True

            else:
                if left_button_held:
                    pyautogui.mouseUp()  # Release the left mouse button
                    left_button_held = False

            if left_button_held:
                left_clicking_overlay = True

            if abs(middle_x - thumb_x) < click_collision_radius and abs(middle_y - thumb_y) < click_collision_radius:
                pyautogui.rightClick()  # Perform a right-click
                right_click_overlay = True

            pyautogui.moveTo(pinky_x, pinky_y)  # Move the mouse cursor to the pinky finger position

            if (
                abs(index_x - middle_x) < click_collision_radius and
                abs(middle_x - pinky_x) < click_collision_radius and
                thumb_y > index_y
            ):
                countdown_start = True  # Start the countdown
            else:
                countdown_start = False
                countdown_timer = 0  # Reset the countdown timer

    if countdown_start:
        if countdown_timer < countdown_duration:
            countdown_text = "Closing: {}".format(countdown_duration - countdown_timer)  # Generate countdown text
            cv2.putText(frame, countdown_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Draw countdown text on the frame
            countdown_timer += 1  # Increment the countdown timer
        else:
            break  # Break out of the loop when the countdown is complete

    if left_click_overlay or left_clicking_overlay:
        cv2.putText(frame, 'Left Clicking', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Draw left click overlay text
        left_click_overlay = False
        left_clicking_overlay = False

    if right_click_overlay:
        cv2.putText(frame, 'Right Click', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Draw right click overlay text
        right_click_overlay = False

    cv2.imshow('Virtual Mouse', frame)  # Display the frame with overlays
    if cv2.waitKey(1) == ord('q'):
        break  # Break the loop if 'q' key is pressed

cap.release()  # Release the camera
cv2.destroyAllWindows()  # Close all OpenCV windows
