import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

# Initialize mouse controller
mouse = Controller()

# Constants for color detection
THRESHOLD = 40
HSV_RED_LOWER = np.array([0, 100, 100])  # lower bound for red/orange
HSV_RED_UPPER = np.array([20, 255, 255])
HSV_BLUE_LOWER = np.array([90, 100, 100])
HSV_BLUE_UPPER = np.array([120, 255, 255])
HSV_YELLOW_LOWER = np.array([20, 100, 100])
HSV_YELLOW_UPPER = np.array([40, 255, 255])
HSV_GREEN_LOWER = np.array([40, 40, 40])
HSV_GREEN_UPPER = np.array([80, 255, 255])
threshold_mod = cv2.THRESH_BINARY

# Initialize variables for mouse control
previous_fingertip = None
left_clicked = False
right_clicked = False

# Create wxPython application
app = wx.App(False)

# Create a window for displaying the webcam feed
frame = wx.Frame(None, wx.ID_ANY, "Webcam Feed")
frame.SetSize(640, 480)
frame.SetBackgroundColour(wx.BLACK)
panel = wx.Panel(frame, wx.ID_ANY)
video_feed = wx.StaticBitmap(panel, wx.ID_ANY)
sizer = wx.BoxSizer(wx.VERTICAL)
sizer.Add(video_feed, 1, wx.EXPAND)
panel.SetSizerAndFit(sizer)
frame.Show()

# Initialize webcam
capture = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = capture.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect color regions
    red_mask = cv2.inRange(hsv_frame, HSV_RED_LOWER, HSV_RED_UPPER)
    blue_mask = cv2.inRange(hsv_frame, HSV_BLUE_LOWER, HSV_BLUE_UPPER)
    yellow_mask = cv2.inRange(hsv_frame, HSV_YELLOW_LOWER, HSV_YELLOW_UPPER)
    green_mask = cv2.inRange(hsv_frame, HSV_GREEN_LOWER, HSV_GREEN_UPPER)

    # Apply adaptive thresholding
    _, red_mask = cv2.threshold(red_mask, THRESHOLD, 255, threshold_mod)
    _, blue_mask = cv2.threshold(blue_mask, THRESHOLD, 255, threshold_mod)
    _, yellow_mask = cv2.threshold(yellow_mask, THRESHOLD, 255, threshold_mod)
    _, green_mask = cv2.threshold(green_mask, THRESHOLD, 255, threshold_mod)

    # Find contours in color masks
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on size
    # contour area condition
    def contour_condition(contour):
        return cv2.contourArea(contour) > 100 and cv2.contourArea(contour) < 1500
    
    filtered_red_contours = [contour for contour in contours_red if contour_condition(contour)]
    filtered_blue_contours = [contour for contour in contours_blue if contour_condition(contour)]
    filtered_yellow_contours = [contour for contour in contours_yellow if contour_condition(contour)]
    filtered_green_contours = [contour for contour in contours_green if contour_condition(contour)]

    # Find bounding boxes for largest contours
    red_bounding_box = cv2.boundingRect(max(filtered_red_contours, key=cv2.contourArea)) if filtered_red_contours else None
    blue_bounding_box = cv2.boundingRect(max(filtered_blue_contours, key=cv2.contourArea)) if filtered_blue_contours else None
    yellow_bounding_box = cv2.boundingRect(max(filtered_yellow_contours, key=cv2.contourArea)) if filtered_yellow_contours else None
    green_bounding_box = cv2.boundingRect(max(filtered_green_contours, key=cv2.contourArea)) if filtered_green_contours else None

    # Perform mouse control based on bounding box positions
    mouse_sensitivity = 2
    if green_bounding_box is not None:
        # mouse.position = (green_bounding_box[0] + green_bounding_box[2] // 2, green_bounding_box[1] + green_bounding_box[3] // 2)
        mouse.position = ((green_bounding_box[0] + green_bounding_box[2] // mouse_sensitivity), mouse_sensitivity * (green_bounding_box[1] + green_bounding_box[3] // mouse_sensitivity))


    if red_bounding_box is not None and blue_bounding_box is not None:
        if red_bounding_box[0] <= blue_bounding_box[0] + blue_bounding_box[2] and \
                blue_bounding_box[0] <= red_bounding_box[0] + red_bounding_box[2] and \
                red_bounding_box[1] <= blue_bounding_box[1] + blue_bounding_box[3] and \
                blue_bounding_box[1] <= red_bounding_box[1] + red_bounding_box[3]:
            if not left_clicked:
                mouse.press(Button.left)
                left_clicked = True
        else:
            if left_clicked:
                mouse.release(Button.left)
                left_clicked = False

    if red_bounding_box is not None and yellow_bounding_box is not None:
        if red_bounding_box[0] <= yellow_bounding_box[0] + yellow_bounding_box[2] and \
                yellow_bounding_box[0] <= red_bounding_box[0] + red_bounding_box[2] and \
                red_bounding_box[1] <= yellow_bounding_box[1] + yellow_bounding_box[3] and \
                yellow_bounding_box[1] <= red_bounding_box[1] + red_bounding_box[3]:
            if not right_clicked:
                mouse.press(Button.right)
                right_clicked = True
        else:
            if right_clicked:
                mouse.release(Button.right)
                right_clicked = False

    # Draw bounding boxes on the frame
    if red_bounding_box is not None:
        cv2.rectangle(frame, (red_bounding_box[0], red_bounding_box[1]),
                      (red_bounding_box[0] + red_bounding_box[2], red_bounding_box[1] + red_bounding_box[3]),
                      (0, 0, 255), 2)
    if blue_bounding_box is not None:
        cv2.rectangle(frame, (blue_bounding_box[0], blue_bounding_box[1]),
                      (blue_bounding_box[0] + blue_bounding_box[2], blue_bounding_box[1] + blue_bounding_box[3]),
                      (255, 0, 0), 2)
    if yellow_bounding_box is not None:
        cv2.rectangle(frame, (yellow_bounding_box[0], yellow_bounding_box[1]),
                      (yellow_bounding_box[0] + yellow_bounding_box[2], yellow_bounding_box[1] + yellow_bounding_box[3]),
                      (0, 255, 255), 2)
    if green_bounding_box is not None:
        cv2.rectangle(frame, (green_bounding_box[0], green_bounding_box[1]),
                      (green_bounding_box[0] + green_bounding_box[2], green_bounding_box[1] + green_bounding_box[3]),
                      (0, 255, 0), 2)

    # Display the frame in the wxPython window
    cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = cv2_img.shape[:2]
    wx_img = wx.Bitmap.FromBuffer(w, h, cv2_img)
    video_feed.SetBitmap(wx_img)

    # Check if the application window is closed
    if cv2.waitKey(1) == 27 or frame is None:
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
