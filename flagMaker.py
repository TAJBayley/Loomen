import cv2
import numpy as np
import time

w = 512
lx = 256
ly = 290

k = 10
n = 1.2
m = 1/n

flag = np.zeros((w, w), np.uint8) #---black in grayscale


for x in range(-w+1, 1):
    for y in range(-w+1, 1):
        if y < -m*x - lx - m*ly + k and y > -m*x - lx - m*ly -k:
            flag.itemset((-x, -y),255)

        if y < m*x - lx + m*ly + k and y > m*x - lx + m*ly - k:
            flag.itemset((-x, -y),255)

        

while True:
    cv2.imshow('black', flag)
    
    key = cv2.waitKey(1) & 0xFF

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
