# import the necessary packages
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import numpy as np
import time


# setting up the pi camera:

camera  = PiCamera()
camera.resolution = (512,512)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size = (512,512))

# For MORPH_CLOSE function:

element1 = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
element2 = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
element3 = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
element4 = cv2.getStructuringElement(cv2.MORPH_CROSS,(9,9))

# kernel for closing image:

kernel = np.ones((9,9),np.uint8)

# Main loop

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    # capture frame
    img = frame.array
    height, width, _ = img.shape
    cv2.imshow('img', img)

    # FILTERING PROCESSES:

    # convert to grayscale:

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray image', img_gray)

    # histogram equalisation:

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_hist = clahe.apply(img_gray)
    #cv2.imshow("img_hist", img_hist)

    # gaussian blur:

    img_blur = cv2.GaussianBlur(img_hist, (5, 5), cv2.BORDER_DEFAULT)
    #cv2.imshow("img_blur", img_blur)

    # adaptive thresholding:

    img_adapted = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    #cv2.imshow("img_adapted", img_adapted)

    # close image:

    img_close = cv2.morphologyEx(img_adapted, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("img_close", img_close)

    # invert image:

    img_invert = cv2.bitwise_not(img_close)
    #cv2.imshow("img_invert", img_invert)


    # segment image by cropping edges (not strictly neccessary but may help):

    black = np.zeros((img_invert.shape[0], img_invert.shape[1], 3), np.uint8) #---black in RGB
    black1 = cv2.rectangle(black,(25,10), ((width-20),(height-40)), (255, 255, 255), -1) #---(x1, y1), (x2, y2)
    gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)               #---converting to gray
    ret,b_mask = cv2.threshold(gray,127,255, 0)                 #---converting to binary
    img_seg = cv2.bitwise_and(img_invert,img_invert,mask = b_mask)
    #cv2.imshow("img_seg", img_seg)

    # find contours in the 'inverted open image':

    _, contours, _ = cv2.findContours(img_seg,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, contours, -1, (0,255,0), 1)

    # set these parameters to zero before start of search through contours:

    area_max = 0
    cX_max = 0
    cY_max = 0

    # make a copy of the original image and binarised image:

    img_original = img.copy()
    img_binarized = img_invert.copy()

    # loop over the contours:

    for c in contours:
            area_current = cv2.contourArea(c)
            M = cv2.moments(c)
            if(M["m00"] != 0):
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    #cv2.circle(img_original, (cX, cY), 1, (0, 0, 255), -1)
                    #cv2.circle(img_binarized, (cX, cY), 1, (0, 0, 255), -1)
                    if area_current > area_max:
                            area_max = cv2.contourArea(c)
                            cX_max = cX
                            cY_max = cY
    cv2.circle(img_original, (cX_max, cY_max), 3, (0, 0, 255), -1)
    cv2.circle(img_binarized, (cX_max, cY_max), 3, (0, 0, 255), -1)


    # show the images (binarised and original):

    cv2.imshow("binarised", img_binarized)
    cv2.imshow("original", img_original)

    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
