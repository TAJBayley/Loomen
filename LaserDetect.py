import numpy as np
import cv2
import time

kernel = np.ones((3,3),np.uint8)

while True:
        img = cv2.imread('vein2019_03_11-19_58_20.jpg', 1)
        height, width, _ = img.shape
        


        img_red = img.copy()
        # set blue and green channels to 0
        img_red[:, :, 0] = 0
        #img_red[:, :, 1] = 0

        cv2.imshow('red image', img_red)

        img_gray = cv2.cvtColor(img_red, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray image', img_gray)

        img_blur = cv2.GaussianBlur(img_gray,(5,5),cv2.BORDER_DEFAULT)
        #img_blur = cv2.medianBlur(img_gray, 5)
        
        img_adaptive = cv2.adaptiveThreshold(img_blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,13,0)
        cv2.imshow('adaptive', img_adaptive)

        img_open = cv2.morphologyEx(img_adaptive, cv2.MORPH_OPEN, kernel)
        cv2.imshow("img_open", img_open)

        _, contours, _ = cv2.findContours(img_open,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0,255,0), 1)

        img_circles = img.copy()

        for c in contours:
                area = cv2.contourArea(c)
                # compute the center of the contour
                M = cv2.moments(c)
                if(M["m00"] != 0):
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        if 90 > area > 30:
                                cv2.circle(img_circles, (cX, cY), 3, (0, 0, 255), -1)

        cv2.imshow('circles',img_circles)



        k = cv2.waitKey(1) & 0xFF
        if k == 27:
                break
        
cv2.destroyAllWindows()
