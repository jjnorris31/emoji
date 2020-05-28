import numpy as np
import cv2
import os

from sklearn import svm
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from keras.preprocessing.image import ImageDataGenerator
import scipy.cluster.hierarchy as shc
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import pickle


path = 'data_res'

# opening the trained model
pickle_in = open("model_trained.p", "rb")
model = pickle.load(pickle_in)


images = []
cnn_images = images
class_number = []
folder_names = os.listdir(path)
print("\nNumber of classes detected: {}".format(len(folder_names)))
num_classes = len(folder_names)
print("\nImporting classes...")

# getting the name of the images and storing
for index, name in enumerate(folder_names):
    folderImgPath = path + '/' + str(name)
    folderImages = os.listdir(folderImgPath)
    for imgFileName in folderImages:
        curImg = cv2.imread(folderImgPath + '/' + imgFileName)
        images.append(curImg)
        class_number.append(index)
    print(name, end=" ")

images = np.array(images)
class_number = np.array(class_number)

img = cv2.resize(images[50], (300, 300))
cv2.imshow("preprocessed", img)
cv2.waitKey(0)

# flatting the images
images = images.reshape((images.shape[0], -1))

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(images, class_number, test_size=.2)


# # SVM
# print("\n\nSVM")
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train, y_train)
#
# # model accuracy
# predicted = clf.predict(X_test)
# print(classification_report(y_test, predicted, target_names=folder_names))
# print('Accuracy:', metrics.accuracy_score(y_test, predicted))
#
#
# # cross validation
# cv_scores = cross_val_score(clf, images, class_number, cv=3)
# print('cv_scores mean: {}'.format(np.mean(cv_scores)))
#
# # KNN
# print("\nKNN")
# knn = KNeighborsClassifier(n_neighbors=14)
# knn.fit(X_train, y_train)
# score = knn.score(X_test, y_test)
#
# # model accuracy
# predicted = knn.predict(X_test)
# print(classification_report(y_test, predicted, target_names=folder_names))
# print("Accuracy:", metrics.accuracy_score(y_test, predicted))
#
# # cross validation
# cv_scores = cross_val_score(knn, images, class_number, cv=3)
# print('cv_scores mean: {}'.format(np.mean(cv_scores)))

# CNN

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(cnn_images, class_number, test_size=.2)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=.2)


def preprocessing(img_process):
    img_process = cv2.cvtColor(img_process, cv2.COLOR_BGR2GRAY)
    img_process = cv2.equalizeHist(img_process)
    img_process = img_process / 255
    return img_process


images_processed = np.array(list(map(preprocessing, cnn_images)))
X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_validation = np.array(list(map(preprocessing, X_validation)))

# # unsupervised learning
# model_unsupervised = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
# plt.figure(figsize=(10, 7))
# plt.title("Emojis Dendograms")
# dend = shc.dendrogram(shc.linkage(model_unsupervised, method='ward'))
# plt.show()
# cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
# cluster.fit_predict(model_unsupervised)
# plt.figure(figsize=(10, 7))
# plt.scatter(model_unsupervised[:, 0], model_unsupervised[:, 1], c=cluster.labels_, cmap='rainbow')
# plt.show()
# cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
# cluster.fit_predict(model_unsupervised)
# plt.figure(figsize=(10, 7))
# plt.scatter(model_unsupervised[:, 0], model_unsupervised[:, 1], c=cluster.labels_, cmap='rainbow')
# plt.show()

# reshaping
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

# move, rotating and more transformations for the image set
dataGen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1,
                             rotation_range=10)

# categorize the data
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_validation = to_categorical(y_validation)
dataGen.fit(X_train)


# # evaluate a model using k-fold cross-validation
# def evaluate_model(data_x, data_y, n_folds=5):
#     scores, histories = list(), list()
#     # prepare cross validation
#     kfold = KFold(n_folds, shuffle=True, random_state=1)
#     # enumerate splits
#     for train_ix, test_ix in kfold.split(dataX):
#         trainX, trainY, testX, testY = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]
#         # fit model
#         history_k = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
#         # evaluate model
#         _, acc = model.evaluate(testX, testY, verbose=0)
#         print('> %.3f' % (acc * 100.0))
#         # stores scores
#         scores.append(acc)
#         histories.append(history_k)
#     return scores, histories


# scores, histories = evaluate_model(X_train, y_train)
# print(scores)
# preprocessing(histories)

#
batchSizeVal = 50
epochsVal = 10
stepsPerEpoch = 2000
#


def my_model():
    num_filters = 60
    size_filter_one = (5, 5)
    size_filter_two = (3, 3)
    size_pool = (2, 2)
    num_node = 500

    model = Sequential()
    model.add((Conv2D(num_filters, size_filter_one,
                      input_shape=(32, 32, 1), activation='relu')))
    model.add((Conv2D(num_filters, size_filter_one, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add((Conv2D(num_filters // 2, size_filter_two, activation='relu')))
    model.add((Conv2D(num_filters // 2, size_filter_two, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(num_node, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = my_model()
print(model.summary())

history = model.fit_generator(dataGen.flow(X_train, y_train, batch_size=batchSizeVal),
                              steps_per_epoch=stepsPerEpoch, epochs=epochsVal,
                              validation_data=(X_validation, y_validation), shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score = ', score[0])
print('Test accuracy', score[1])

# saving the file of our model trained
pickle_out = open("model_trained.p", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()
