import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx

def leftClick():
    mouse = Controller()

    app = wx.App(False)
    (sx, sy) = wx.GetDisplaySize()
    (camx, camy) = (480, 320)

    lower_bound = np.array([130, 90, 80])
    upper_bound = np.array([203, 225, 180])

    cam = cv2.VideoCapture(0)
    cam.set(3, camx)
    cam.set(4, camy)

    kernelOpen = np.ones((5, 5))
    kernelClose = np.ones((20, 20))

    mLocOld = np.array([0, 0])
    mouseLoc = np.array([0, 0])
    Df = 2.3
    pinchFlag = 0

    min_area_threshold = 200  # Minimum area threshold for red color detection

    while True:
        ret, img = cam.read()
        img = cv2.flip(img, 1)

        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(imgHSV, lower_bound, upper_bound)
        maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
        maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
        maskFinal = maskClose

        conts, h = cv2.findContours(maskFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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

        cv2.imshow('Cam', img)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    leftClick()
