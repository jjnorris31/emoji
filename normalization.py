import cv2
import numpy as np
import os

path = 'data'
folder_names = ['Angry', 'Happy', 'Poo', 'Sad', 'Surprised']
images = []


for name in folder_names:
    folderImgPath = path + '/' + name
    folderImages = os.listdir(folderImgPath)
    for imgFileName in folderImages:
        curImg = cv2.imread(folderImgPath + '/' + imgFileName)

        # getting an image in grayscale and threshold
        gray_img = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)
        ret, thresh_img = cv2.threshold(gray_img, 127, 255, 0)

        # getting the contours
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # getting the largest area of the contours
        largest_area = 0
        largest_index = 0
        for i in range(1, len(contours)):
            area = cv2.contourArea(contours[i])
            if area > largest_area:
                largest_area = area
                largest_index = i

        # image binarization
        cnt = contours[largest_index]
        x, y, w, h = cv2.boundingRect(cnt)
        newImg = curImg[y:y + h, x:x + w]

        # bin the image
        ret, newImg = cv2.threshold(newImg, 127, 255, cv2.THRESH_BINARY_INV)

        # dilating the borders of emoji image
        kernel = np.ones((20, 20), np.uint8)
        dilate = cv2.dilate(newImg, kernel, iterations=1)

        newPath = 'C:\\Users\\jjnor\\PycharmProjects\\emoji\\data_res\\' + name
        print(os.path.join(newPath, imgFileName))
        print(cv2.imwrite(os.path.join(newPath, imgFileName), cv2.resize(dilate, (32, 32), cv2.INTER_LANCZOS4)))


