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
threshold_mod = cv2.THRESH_OTSU + cv2.THRESH_BINARY

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
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contour = max(contours, key=cv2.contourArea) if contours else None

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contour = max(contours, key=cv2.contourArea) if contours else None

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contour = max(contours, key=cv2.contourArea) if contours else None

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contour = max(contours, key=cv2.contourArea) if contours else None

    # Draw contours on the frame
    if red_contour is not None:
        cv2.drawContours(frame, [red_contour], -1, (0, 0, 255), 2)
        red_fingertip = tuple(red_contour[red_contour[:, :, 1].argmax()][0])
    else:
        red_fingertip = None

    if blue_contour is not None:
        cv2.drawContours(frame, [blue_contour], -1, (255, 0, 0), 2)
        blue_fingertip = tuple(blue_contour[blue_contour[:, :, 1].argmax()][0])
    else:
        blue_fingertip = None

    if yellow_contour is not None:
        cv2.drawContours(frame, [yellow_contour], -1, (0, 255, 255), 2)
        yellow_fingertip = tuple(yellow_contour[yellow_contour[:, :, 1].argmax()][0])
    else:
        yellow_fingertip = None

    if green_contour is not None:
        cv2.drawContours(frame, [green_contour], -1, (0, 255, 0), 2)
        green_fingertip = tuple(green_contour[green_contour[:, :, 1].argmax()][0])
    else:
        green_fingertip = None

    # Perform mouse control based on finger positions
    if green_fingertip is not None:
        mouse.position = (green_fingertip[0] * 2, green_fingertip[1] * 2)  # Adjust mouse position based on webcam resolution

    if red_fingertip is not None and blue_fingertip is not None:
        if abs(red_fingertip[0] - blue_fingertip[0]) < 20 and abs(red_fingertip[1] - blue_fingertip[1]) < 20:
            if not left_clicked:
                mouse.press(Button.left)
                left_clicked = True
        else:
            if left_clicked:
                mouse.release(Button.left)
                left_clicked = False

    if red_fingertip is not None and yellow_fingertip is not None:
        if abs(red_fingertip[0] - yellow_fingertip[0]) < 20 and abs(red_fingertip[1] - yellow_fingertip[1]) < 20:
            if not right_clicked:
                mouse.press(Button.right)
                right_clicked = True
        else:
            if right_clicked:
                mouse.release(Button.right)
                right_clicked = False

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
