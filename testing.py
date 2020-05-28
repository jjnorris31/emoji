import numpy as np
import cv2
import pickle
from tkinter import *
from PIL import ImageTk, Image, ImageDraw
import PIL

#
width = 480
height = 480
center = height//2
white = (255, 255, 255)
green = (0, 128, 0)
#

# opening the trained model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)


def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)  # normalize
    img = img / 255
    return img


def save():
    filename = "image.png"
    image1.save(filename)
    root.destroy()


def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black", width=15)
    draw.line([x1, y1, x2, y2], fill="black", width=30)


root = Tk()

# Tkinter create a canvas to draw on
cv = Canvas(root, width=width, height=height, bg='white')
cv.pack()

# PIL create an image
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)


cv.pack(expand=YES, fill=BOTH)
cv.bind("<B1-Motion>", paint)
button = Button(text="save", command=save)
button.pack()
root.mainloop()


def get_class_name(num_class):
    classes = {
        0: "Angry",
        1: "Happy",
        2: "Poo",
        3: "Sad",
        4: "Surprised"
    }
    return classes.get(num_class, "?")


def test():
    img = cv2.imread("image.png")

    # first pre processing
    ret, bin_image = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((20, 20), np.uint8)
    dilate = cv2.dilate(bin_image, kernel, iterations=1)

    # second processing
    test_img = cv2.resize(dilate, (32, 32))
    test_img = preprocessing(test_img)
    test_img = test_img.reshape(1, 32, 32, 1)

    # testing with the model
    class_index = int(model.predict_classes(test_img))
    predictions = model.predict(test_img)
    prob_value = np.amax(predictions)
    print(get_class_name(class_index), prob_value)


test()


