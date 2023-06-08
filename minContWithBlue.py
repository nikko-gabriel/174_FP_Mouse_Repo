import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

def leftClick():
    mouse = Controller()

    app = wx.App(False)
    (sx, sy) = wx.GetDisplaySize()
    (camx, camy) = (480, 320)

    red_lower_bound = np.array([130, 90, 80])
    red_upper_bound = np.array([203, 225, 180])

    blue_lower_bound = np.array([90, 60, 0])
    blue_upper_bound = np.array([140, 255, 255])

    cam = cv2.VideoCapture(0)
    cam.set(3, camx)
    cam.set(4, camy)

    kernelOpen = np.ones((5, 5))
    kernelClose = np.ones((20, 20))

    mLocOld = np.array([0, 0])
    mouseLoc = np.array([0, 0])
    Df = 2.3
    pinchFlag = 0

    min_area_threshold = 200  # Minimum area threshold for color detection

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)

        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(imgHSV, red_lower_bound, red_upper_bound)
        blue_mask = cv2.inRange(imgHSV, blue_lower_bound, blue_upper_bound)

        red_mask_open = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernelOpen)
        red_mask_close = cv2.morphologyEx(red_mask_open, cv2.MORPH_CLOSE, kernelClose)

        blue_mask_open = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernelOpen)
        blue_mask_close = cv2.morphologyEx(blue_mask_open, cv2.MORPH_CLOSE, kernelClose)

        mask_final = red_mask_close

        conts, h = cv2.findContours(mask_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(conts) >= 2:
            if pinchFlag == 1:
                pinchFlag = 0
                mouse.release(Button.left)

            for i in range(2):
                x, y, w, h = cv2.boundingRect(conts[i])
                area = cv2.contourArea(conts[i])

                if area > min_area_threshold:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Remaining code...
        elif len(conts) == 1:
            x, y, w, h = cv2.boundingRect(conts[0])
            area = cv2.contourArea(conts[0])

            if area > min_area_threshold:
                if pinchFlag == 0:
                    pinchFlag = 1
                    mouse.press(Button.left)

                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Remaining code...

        blue_conts, _ = cv2.findContours(blue_mask_close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(conts) >= 1 and len(blue_conts) >= 1:
            for i in range(len(conts)):
                for j in range(len(blue_conts)):
                    red_x, red_y, red_w, red_h = cv2.boundingRect(conts[i])
                    blue_x, blue_y, blue_w, blue_h = cv2.boundingRect(blue_conts[j])

                    if abs((red_x + red_w / 2) - (blue_x + blue_w / 2)) < max(red_w, blue_w) / 2 and abs(
                            (red_y + red_h / 2) - (blue_y + blue_h / 2)) < max(red_h, blue_h) / 2:
                        mouse.click(Button.left, 1)

        cv2.imshow('Cam', img)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    leftClick()
