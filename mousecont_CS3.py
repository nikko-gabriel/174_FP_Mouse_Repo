import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

# Initialize mouse controller
mouse = Controller()

# Constants for color detection
THRESHOLD = 40
HSV_RED_LOWER = np.array([0, 100, 100])
HSV_RED_UPPER = np.array([20, 255, 255])
HSV_BLUE_LOWER = np.array([90, 100, 100])
HSV_BLUE_UPPER = np.array([120, 255, 255])
HSV_YELLOW_LOWER = np.array([20, 100, 100])
HSV_YELLOW_UPPER = np.array([40, 255, 255])
HSV_GREEN_LOWER = np.array([60, 100, 100])
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
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contour = max(contours, key=cv2.contourArea) if contours and cv2.contourArea(contours[0]) > 300 else None

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_contour = max(contours, key=cv2.contourArea) if contours and cv2.contourArea(contours[0]) > 300 else None

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contour = max(contours, key=cv2.contourArea) if contours and cv2.contourArea(contours[0]) > 300 else None

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contour = max(contours, key=cv2.contourArea) if contours and cv2.contourArea(contours[0]) > 300 else None

    # Draw bounding boxes on the frame
    if red_contour is not None:
        x_red, y_red, w_red, h_red = cv2.boundingRect(red_contour)
        cv2.rectangle(frame, (x_red, y_red), (x_red + w_red, y_red + h_red), (0, 0, 255), 2)

    if blue_contour is not None:
        x_blue, y_blue, w_blue, h_blue = cv2.boundingRect(blue_contour)
        cv2.rectangle(frame, (x_blue, y_blue), (x_blue + w_blue, y_blue + h_blue), (255, 0, 0), 2)

    if yellow_contour is not None:
        x_yellow, y_yellow, w_yellow, h_yellow = cv2.boundingRect(yellow_contour)
        cv2.rectangle(frame, (x_yellow, y_yellow), (x_yellow + w_yellow, y_yellow + h_yellow), (0, 255, 255), 2)

    if green_contour is not None:
        x_green, y_green, w_green, h_green = cv2.boundingRect(green_contour)
        cv2.rectangle(frame, (x_green, y_green), (x_green + w_green, y_green + h_green), (0, 255, 0), 2)

    # Perform mouse control based on bounding box positions
    if green_contour is not None:
        x_center = x_green + (w_green // 2)
        y_center = y_green + (h_green // 2)
        screen_size = wx.GetDisplaySize()
        screen_width, screen_height = screen_size[0], screen_size[1]
        mouse_x = (x_center * screen_width) // frame.shape[1]
        mouse_y = (y_center * screen_height) // frame.shape[0]
        mouse.position = (mouse_x, mouse_y)

    if red_contour is not None and blue_contour is not None:
        if abs((x_red + (w_red // 2)) - (x_blue + (w_blue // 2))) < 20 and abs((y_red + (h_red // 2)) - (y_blue + (h_blue // 2))) < 20:
            if not left_clicked:
                mouse.press(Button.left)
                left_clicked = True
        else:
            if left_clicked:
                mouse.release(Button.left)
                left_clicked = False

    if red_contour is not None and yellow_contour is not None:
        if abs((x_red + (w_red // 2)) - (x_yellow + (w_yellow // 2))) < 20 and abs((y_red + (h_red // 2)) - (y_yellow + (h_yellow // 2))) < 20:
            if not right_clicked:
                mouse.press(Button.right)
                right_clicked = True
        else:
            if right_clicked:
                mouse.release(Button.right)
                right_clicked = False

    # Convert the frame to RGB for wxPython display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert numpy array to wxImage
    img = wx.Image(frame_rgb.shape[1], frame_rgb.shape[0])
    img.SetData(frame_rgb.tobytes())

    # Convert wxImage to wxBitmap for display
    bmp = wx.Bitmap(img)

    # Show the bitmap on the wxPython window
    video_feed.SetBitmap(bmp)

    # Check if the application window is closed
    if cv2.waitKey(1) == 27 or frame is None:
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
