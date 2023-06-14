import cv2
import numpy as np
import wx
from pynput.mouse import Button, Controller

# Initialize mouse controller
mouse = Controller()

# Constants for color ranges in HSV
LOWER_RED_ORANGE = np.array([0, 100, 100], dtype=np.uint8)
UPPER_RED_ORANGE = np.array([20, 255, 255], dtype=np.uint8)
LOWER_BLUE = np.array([90, 100, 100], dtype=np.uint8)
UPPER_BLUE = np.array([120, 255, 255], dtype=np.uint8)
LOWER_YELLOW = np.array([20, 100, 100], dtype=np.uint8)
UPPER_YELLOW = np.array([40, 255, 255], dtype=np.uint8)
LOWER_GREEN = np.array([60, 100, 100], dtype=np.uint8)
UPPER_GREEN = np.array([80, 255, 255], dtype=np.uint8)

# Create wxPython application object
app = wx.App()

# Create a window to display the webcam feed
frame = wx.Frame(None, -1, 'Webcam Feed')
frame.SetDimensions(0, 0, 800, 600)
frame.SetBackgroundColour(wx.BLACK)
frame.Show()

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame_cv = cap.read()
    
    if ret:
        # Flip the frame horizontally
        frame_cv = cv2.flip(frame_cv, 1)
        
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2HSV)
        
        # Threshold the colors
        mask_red_orange = cv2.inRange(hsv, LOWER_RED_ORANGE, UPPER_RED_ORANGE)
        mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
        mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
        mask_green = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        
        # Find contours in each mask
        contours_red_orange, _ = cv2.findContours(mask_red_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour for each color
        max_contour_red_orange = max(contours_red_orange, key=cv2.contourArea) if contours_red_orange else None
        max_contour_blue = max(contours_blue, key=cv2.contourArea) if contours_blue else None
        max_contour_yellow = max(contours_yellow, key=cv2.contourArea) if contours_yellow else None
        max_contour_green = max(contours_green, key=cv2.contourArea) if contours_green else None
        
        # Get bounding boxes for the largest contours
        x_red_orange, y_red_orange, w_red_orange, h_red_orange = cv2.boundingRect(max_contour_red_orange) if max_contour_red_orange is not None else (0, 0, 0, 0)
        x_blue, y_blue, w_blue, h_blue = cv2.boundingRect(max_contour_blue) if max_contour_blue is not None else (0, 0, 0, 0)
        x_yellow, y_yellow, w_yellow, h_yellow = cv2.boundingRect(max_contour_yellow) if max_contour_yellow is not None else (0, 0, 0, 0)
        x_green, y_green, w_green, h_green = cv2.boundingRect(max_contour_green) if max_contour_green is not None else (0, 0, 0, 0)
        
        # Draw bounding boxes on the frame
        cv2.rectangle(frame_cv, (x_red_orange, y_red_orange), (x_red_orange + w_red_orange, y_red_orange + h_red_orange), (0, 0, 255), 2)
        cv2.rectangle(frame_cv, (x_blue, y_blue), (x_blue + w_blue, y_blue + h_blue), (255, 0, 0), 2)
        cv2.rectangle(frame_cv, (x_yellow, y_yellow), (x_yellow + w_yellow, y_yellow + h_yellow), (0, 255, 255), 2)
        cv2.rectangle(frame_cv, (x_green, y_green), (x_green + w_green, y_green + h_green), (0, 255, 0), 2)
        
        # Perform actions based on the position of colors
        if x_red_orange > 0 and y_green > 0:
            # Move the mouse based on green (pinky) position
            mouse.position = (x_green, y_green)
        
        if x_red_orange > 0 and x_blue > 0:
            # Perform left click if red/orange (thumb) and blue (index) touch
            mouse.press(Button.left)
            mouse.release(Button.left)
        
        if x_red_orange > 0 and x_yellow > 0:
            # Perform right click if red/orange (thumb) and yellow (middle) touch
            mouse.press(Button.right)
            mouse.release(Button.right)
        
        # Convert the frame to RGB for wxPython display
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to wxImage
        img = wx.Image(frame_rgb.shape[1], frame_rgb.shape[0])
        img.SetData(frame_rgb.tobytes())
        
        # Convert wxImage to wxBitmap for display
        bmp = wx.Bitmap(img)
        
        # Create a wxPython static bitmap and display the webcam feed
        static_bitmap = wx.StaticBitmap(frame, -1, bmp)
        
        # Refresh the window
        frame.Refresh()
        
        # Check for key press and exit the loop if 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

# Release the webcam and destroy windows
cap.release()
cv2.destroyAllWindows()
