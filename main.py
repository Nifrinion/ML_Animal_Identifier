import sys

import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from tensorflow.keras import datasets
from tensorflow.keras import layers
from tensorflow.keras import models
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
from PyQt5.QtGui import QPixmap, QImage


filepaths = []
current_image_index = 0
total_images = 0
input_folder = ""
output_folder = ""


class MyGui(QMainWindow):

    def __init__(self):
        super(MyGui, self).__init__()
        uic.loadUi("projectUI_1.2.ui", self)

        self.show()

        self.imageStatus.setText("0/0")
        self.thumbnailStatus.setText("0/0")
        self.scanProgressBar.setValue(0)

        self.outputFolderButton.clicked.connect(self.say)
        self.loadImageButton.clicked.connect(self.load_images_from_folder)
        self.scanImagesButton.clicked.connect(self.say)
        self.prevImageButton.clicked.connect(self.prev_image)
        self.nextImageButton.clicked.connect(self.next_image)

    def say(self):
        print("Hey")

    def load_images_from_folder(self):
        global filepaths
        global total_images
        filepaths.clear()
        folder_path = filedialog.askdirectory()
        if folder_path:
            file_names = os.listdir(folder_path)
            filepaths = [os.path.join(folder_path, file) for file in file_names if
                         file.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if filepaths:
                total_images = len(filepaths)
                print(f"Total images: {total_images}")
                self.display_image()
            else:
                print("No images in selected folder")

    def next_image(self):
        global current_image_index
        current_image_index = (current_image_index + 1) % len(filepaths)
        self.display_image()

    def prev_image(self):
        global current_image_index
        current_image_index = (current_image_index - 1) % len(filepaths)
        self.display_image()

    def display_image(self):
        self.imageStatus.setText(str(current_image_index+1)+"/"+str(total_images))
        if filepaths and current_image_index < len(filepaths):
            image_original = Image.open(filepaths[current_image_index])
            image_original.thumbnail((256, 256))
            # convert image to byte array for the QPixmap
            image_byte_array = image_original.convert("RGBA").tobytes("raw", "RGBA")
            pixmap = QtGui.QPixmap.fromImage(
                QtGui.QImage(image_byte_array, image_original.size[0], image_original.size[1],
                             QtGui.QImage.Format_RGBA8888))
            self.fullImageLabel.setPixmap(pixmap)
            if current_image_index == 0:
                self.scanFullImageLabel.setPixmap(pixmap)
        else:
            print("No image to display or index out of range")


def load_model():
    dataset_path = r'C:\Users\User\Desktop\Dataset\Images'

    images = []
    labels = []

    for label, category in enumerate(['Cats', 'Dogs']):
        category_path = os.path.join(dataset_path, category)
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name)
            # Load and resize the image to 32x32
            image = cv.imread(image_path)
            image = cv.resize(image, (32, 32))
            images.append(image)
            labels.append(label)  # Assign label 0 for cats and label 1 for dogs

    # Convert lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    images = images / 255

    # defining out identifiable categories
    class_names = ['Cat', 'Dog']
    model = models.load_model('image_classifier.model')

    folder_path = r'C:\Users\User\Desktop\Animals'
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv.imread(img_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (32, 32))

            plt.imshow(img, cmap=plt.cm.binary)

            # obtain class probabilities
            prediction = model.predict(np.array([img]) / 255)
            # get index for prediction
            index = np.argmax(prediction)
            # get the max probability of the class probabilities; aka of our predicted class
            confidence_score = np.max(prediction)

            print(f'Prediction: {class_names[index]}, Confidence Score: {confidence_score}')


def create_model():
    print("Creating Model....")
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Use a validation split of 0.2 (20% of the data)
    history = model.fit(images, labels, epochs=10, validation_split=0.2)

    # Get the loss and accuracy metrics from the training process
    train_loss = history.history['loss'][-1]
    train_accuracy = history.history['accuracy'][-1]

    # Get the loss and accuracy metrics from the validation process
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1]

    # lower is better
    print(f"Final Training Loss: {train_loss}")
    # higher is better
    print(f"Final Training Accuracy: {train_accuracy}")
    # lower is better
    print(f"Final Validation Loss: {val_loss}")
    # higher is better
    print(f"Final Validation Accuracy: {val_accuracy}")


def close_program():
    # make sure to change this later; you dont want to close the program mid process ...
    sys.exit(0)


app = QApplication([])
window = MyGui()
app.exec_()

load_model()




