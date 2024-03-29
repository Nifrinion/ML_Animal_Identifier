# Andrew Cornish; WGU; C964
#
# imports
import math
import shutil
import sys
import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tkinter as tk
from tkinter import filedialog
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow
import time
from PIL import ImageTk, Image


#global variables
filepaths = []
current_image_index = 0
image_index = 0
total_images = 1
input_folder = r'C:\Users\User\PycharmProjects\pythonProject1\InputData'
output_folder = r'C:\Users\User\PycharmProjects\pythonProject1\OutputData'
dataset_path = r'C:\Users\User\Desktop\Dataset\Images'
images = []
labels = []
class_names = ['Bears', 'Cats']
image_data_list = []
model = models.load_model('image_classifier.model')
model_name = 'image_classifier'
confidence_threshold = 0


# App Class
class MyGui(QMainWindow):

    #   initialization function
    def __init__(self):
        super(MyGui, self).__init__()

        # load UI file
        uic.loadUi("projectUI_1.2.ui", self)
        self.show()

        self.closeEvent = self.close_program

        # initialize UI elements and connect with functions
        self.setWindowIcon(QIcon('cat.jpg'))
        self.setWindowTitle("ML Animal Identifier")
        self.imageStatus.setText("0/0")
        self.thumbnailStatus.setText("0/0")
        self.outputEntry.setText(output_folder)
        self.loadModelEntry.setText(model_name)
        self.outputFolderButton.clicked.connect(self.load_output_folder)
        self.loadImageButton.clicked.connect(self.load_images_from_folder)
        self.scanImagesButton.clicked.connect(self.scan_images)
        self.imageSlider.valueChanged.connect(self.adjust_image)
        self.confidenceSlider.valueChanged.connect(self.adjust_threshold)
        self.trainModelButton.clicked.connect(self.create_model)

        # create and set status bar elements
        global progress_bar
        progress_bar = QProgressBar()
        progress_bar.setMinimum(0)
        progress_bar.setMaximum(100)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.statusbar.addWidget(spacer)
        self.statusbar.addWidget(progress_bar)
        self.statusbar.setStyleSheet("background-color: #BF4E30;")

        progress_bar.setValue(0)
        self.statusbar.showMessage("Ready")

    # function for creating image data dictionary object
    def create_image_info(self, image_no, classification, confidence_score, file_location):
        image_info = {
            "image_no": image_no,
            "classification": classification,
            "confidence_score": confidence_score,
            "file_location": file_location
        }
        return image_info

    # loads folder path from input folder
    def load_images_from_folder(self):
        global filepaths
        global total_images
        global input_folder
        global current_image_index
        global image_index

        self.imageScannerGroupbox.setEnabled(False)
        self.imageInputGroupbox.setEnabled(False)
        self.imageViewerGroupbox.setEnabled(False)
        current_image_index = 0
        image_index = 0
        total_images = 0

        filepaths.clear()
        # receives folder path from user input
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.loadImagesEntry.setText(folder_path)
            input_folder = folder_path
            file_names = os.listdir(folder_path)
            filepaths = [os.path.join(folder_path, file) for file in file_names if
                         file.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if filepaths:
                total_images = len(filepaths)
                print(f"Total images: {total_images}")
                self.scanImagesButton.setEnabled(True)
                self.confidenceSlider.setEnabled(True)
                self.confidenceLabel.setEnabled(True)
                self.imageSlider.setMaximum(total_images)
            else:
                print("No images in selected folder")

    # allows user to adjust which image/image data they are viewing from the input set
    def adjust_image(self):
        global current_image_index
        current_image_index = self.imageSlider.value() - 1
        self.display_image()

    # allows user to decide what confidence score is acceptable
    def adjust_threshold(self):
        global confidence_threshold
        confidence_threshold = self.confidenceSlider.value()
        self.confidenceLabel.setText(f'Confidence Threshold: {confidence_threshold}')

    # displays the images
    def display_image(self):
        print(current_image_index)
        print(image_index)
        bar_value = math.floor(((image_index + 1) / total_images) * 100)
        progress_bar.setValue(bar_value)
        print(total_images)

        # label data
        self.imageStatus.setText(str(current_image_index+1)+"/"+str(total_images))
        self.imageClass.setText(f'Classification: {image_data_list[current_image_index]["classification"]}')
        self.imageConfidence.setText(f'Confidence: {image_data_list[current_image_index]["confidence_score"]}')
        self.imageLocation.setText(f'File: {image_data_list[current_image_index]["file_location"]}')

        self.thumbnailStatus.setText(str(image_index)+"/"+str(total_images))
        self.thumbnailClass.setText(f'Classification: {image_data_list[image_index-1]["classification"]}')
        self.thumbnailConfidence.setText(f'Confidence: {image_data_list[image_index-1]["confidence_score"]}')

        if filepaths and current_image_index < len(filepaths):
            # image input
            image_scan = Image.open(filepaths[image_index-1])
            image_scan.thumbnail((260, 260))
            image_byte_array_scan = image_scan.convert("RGBA").tobytes("raw", "RGBA")
            pixmap_scan = QtGui.QPixmap.fromImage(
                QtGui.QImage(image_byte_array_scan, image_scan.size[0], image_scan.size[1],
                             QtGui.QImage.Format_RGBA8888))
            self.scanFullImageLabel.setPixmap(pixmap_scan)

            # thumbnail image/ Scan View image
            image_scan.thumbnail((32, 32))
            image_byte_array_scan = image_scan.convert("RGBA").tobytes("raw", "RGBA")
            pixmap_scan = QtGui.QPixmap.fromImage(
                QtGui.QImage(image_byte_array_scan, image_scan.size[0], image_scan.size[1],
                             QtGui.QImage.Format_RGBA8888))
            self.scanThumbnailImageLabel.setPixmap(pixmap_scan)

            # data viewer image
            image_original = Image.open(filepaths[current_image_index])
            image_original.thumbnail((260, 260))
            image_byte_array = image_original.convert("RGBA").tobytes("raw", "RGBA")
            pixmap = QtGui.QPixmap.fromImage(
                QtGui.QImage(image_byte_array, image_original.size[0], image_original.size[1],
                             QtGui.QImage.Format_RGBA8888))
            self.fullImageLabel.setPixmap(pixmap)
        else:
            print("No image to display or index out of range")

    # creates appropriate label data for training
    def load_images_labels(self):
        global dataset_path
        global images
        global labels
        global class_names

        # create labels for images by folder
        for label, category in enumerate(['Bears', 'Cats']):
            category_path = os.path.join(dataset_path, category)
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                # Load and resize the image to 32x32; testing 244x244
                image = cv.imread(image_path)
                image = cv.resize(image, (32, 32))
                images.append(image)
                labels.append(label)  # Assign label 0 for bears and label 1 for cats

        images = np.array(images) / 255
        labels = np.array(labels)
        class_names = ['Bears', 'Cats']

    # loads the saved model
    def load_model(self):
        global model
        model = models.load_model('image_classifier.model')
        return

    # allows user to select the output folder
    def load_output_folder(self):
        global output_folder
        output_folder = filedialog.askdirectory()

    # utilizes the model to scan and identify images from the iunput folder
    def scan_images(self):
        global image_index
        global image_data_list
        global input_folder
        global output_folder
        global class_names
        class_1 = 0
        class_2 = 0

        if self.scanImagesButton.text() == "Stop":

            self.scanImagesButton.setText("Scan")
            self.imageScannerGroupbox.setEnabled(False)
            self.imageInputGroupbox.setEnabled(False)
            self.imageViewerGroupbox.setEnabled(False)
            return

        # create output folders
        if class_names:
            os.makedirs(os.path.join(output_folder, 'Unidentified'), exist_ok=True)
            for class_name in class_names:
                class_folder = os.path.join(output_folder, class_name)
                os.makedirs(class_folder, exist_ok=True)
                print(f"Created folder: {class_folder}")

        folder_path = input_folder
        self.imageScannerGroupbox.setEnabled(True)
        self.imageInputGroupbox.setEnabled(True)
        self.scanImagesButton.setText("Stop")

        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
                image_index = image_index + 1
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

                image = self.create_image_info(image_index, class_names[index], confidence_score, img_path)
                image_data_list.append(image)
                print(image)

                print(f'Prediction: {class_names[index]}, Confidence Score: {confidence_score}')

                # depending on the confidence threshold; place images in output folders
                if confidence_score >= confidence_threshold:
                    shutil.copy(img_path, os.path.join(output_folder, class_names[index]))
                else:
                    shutil.copy(img_path, os.path.join(output_folder, 'Unidentified'))

                # add to class list for later validation checking
                self.display_image()
                if index == 0:
                    class_1 = class_1 + 1
                else:
                    class_2 = class_2 + 1

                # allows for image processing
                QApplication.processEvents()

            # if button is paused; disable/pause
            if self.scanImagesButton.text() == "Scan":
                self.imageScannerGroupbox.setEnabled(False)
                self.imageInputGroupbox.setEnabled(False)
                self.imageViewerGroupbox.setEnabled(False)
                return
        self.imageViewerGroupbox.setEnabled(True)
        self.scanImagesButton.setText("Scan")
        image_index = 0
        progress_bar.setValue(0)

        # print validation data
        print(f"Bear Total: {class_1}")
        print(f"Cat Total: {class_2}")

    def create_model(self):

        print("Loading Images and Labels...")
        self.load_images_labels()

        print("Creating Model....")
        # allows use to add sequential layers
        model = models.Sequential()

        # layer 1; input shape is 32x32 images
        model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))

        # subsequent layers
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # flattens our multidimensional output
        model.add(layers.Flatten())

        # our final layers, ensure 2 possible outputs (class 1, class 2)
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))

        # compiles model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        #datagen = ImageDataGenerator(horizontal_flip=True)
        # Concatenate original and flipped images
        #augmented_images = np.concatenate((images, np.flip(images, axis=2)), axis=0)
        #augmented_labels = np.concatenate((labels, labels), axis=0)

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

        # save model
        model.save('image_classifier.model')

    # closes program
    def close_program(self):
        QApplication.quit()
        sys.exit(0)


# load app/gui
app = QApplication([])
window = MyGui()
window.load_model()
app.exec_()
