import os
import random
import numpy as np
import tensorflow as tf
import mahotas
import cv2
import itertools
import time
import csv

from typing import List, Dict

from tkinter import messagebox

from PIL import Image

import matplotlib.pyplot as plt


class Algorithms:
    BIRADS_CLASSES = ["1", "2", "3", "4"]

    def __init__(self) -> None:
        self.images: Dict[str, List[Image.Image]] = {}

        self.set_number_of_shades_of_gray(32)

        self.set_used_descriptors(
            True, True, True, True, True, True, True,
            True, True, True, True, True, True, True
        )

    @staticmethod
    def get_7_invariant_hu_moments(image):
        all_hu_moments = cv2.moments(image)
        return cv2.HuMoments(all_hu_moments).flatten()

    @staticmethod
    def get_haralick_descriptors(image):
        """
        Angular Second Moment: Energy or Uniformity
        Contrast
        Correlation
        Sum of Squares: Variance
        Inverse Difference Moment: Texture Homogeneity
        Sum Average
        Sum Variance
        Sum Entropy
        Entropy
        Difference Variance
        Difference Entropy
        Information Measures of Correlation
        Information Measures of Correlation

        Left-to-Right
        Top-to-Bottom
        Top Left-to-Bottom Right
        Top Right-to-Bottom Left
        """
        radii = [1, 2, 4, 8, 16]
        num_of_haralick_descriptors = 13
        haralick_descriptors_for_all_radii: np.ndarray = np.empty(
            shape=(len(radii), num_of_haralick_descriptors))

        for i in range(len(radii)):
            haralick_descriptors: np.ndarray = np.array(mahotas.features.haralick(
                image, distance=radii[i]
            ))
            haralick_descriptors = haralick_descriptors.mean(axis=0)  # Mean of each column
            haralick_descriptors_for_all_radii[i] = haralick_descriptors

        return haralick_descriptors_for_all_radii.mean(axis=0)  # Mean of each column

    def get_all_image_descriptors(self, image):
        descriptors = self.get_haralick_descriptors(image)

        descriptors = np.append(
            descriptors, self.get_7_invariant_hu_moments(image), axis=0)

        return descriptors

    def get_image_descriptors(self, image):
        descriptors = self.get_all_image_descriptors(image)

        return np.array([descriptors[i] for i in self.indexes_of_the_used_descriptors])

    def resample_image(self, image):
        return np.round(np.array(image) / self.resample_ratio).astype(np.uint8)

    def set_number_of_shades_of_gray(self, number_of_shades: int):
        self.number_of_shades = number_of_shades
        self.resample_ratio = int(round(255 / number_of_shades))

    def get_training_and_test_set_for_birads_class(self, birads_class):
        images = self.images[birads_class]
        testing_set_size = int(len(images) / 4)

        training_set: np.ndarray = np.empty(
            shape=(len(images), len(self.indexes_of_the_used_descriptors)),
            dtype=np.float64
        )

        for i in range(len(images)):
            image = self.resample_image(images[i])
            training_set[i] = self.get_image_descriptors(image)

        testing_set: np.ndarray = np.empty(
            shape=(testing_set_size, len(self.indexes_of_the_used_descriptors)),
            dtype=np.float64
        )

        for i in range(testing_set_size):
            random_index = random.randint(0, len(training_set) - 1)
            testing_set[i] = training_set[random_index]
            training_set = np.delete(training_set, random_index, axis=0)

        return (training_set, testing_set)

    def get_training_and_test_set(self):
        self.images_training_set: np.ndarray = np.empty(
            shape=(0, len(self.indexes_of_the_used_descriptors)),
            dtype=np.float64
        )
        self.images_training_labels_set = []

        self.images_testing_set: np.ndarray = np.empty(
            shape=(0, len(self.indexes_of_the_used_descriptors)),
            dtype=np.float64
        )
        self.images_testing_labels_set = []

        for birads_class in self.BIRADS_CLASSES:
            training_set, testing_set = self.get_training_and_test_set_for_birads_class(
                birads_class)

            self.images_training_set = np.append(self.images_training_set, training_set, axis=0)
            training_labels = [int(birads_class) - 1] * len(training_set)
            self.images_training_labels_set.extend(training_labels)

            self.images_testing_set = np.append(self.images_testing_set, testing_set, axis=0)
            testing_labels = [int(birads_class) - 1] * len(testing_set)
            self.images_testing_labels_set.extend(testing_labels)

        self.images_testing_labels_set: np.ndarray = np.array(
            self.images_testing_labels_set
        )
        self.images_training_labels_set: np.ndarray = np.array(
            self.images_training_labels_set
        )

    def set_used_descriptors(
        self,
        energyCheck: bool,
        contrastCheck: bool,
        correlationCheck: bool,
        varianceCheck: bool,
        homogeneityCheck: bool,
        sumAverageCheck: bool,
        sumVarianceCheck: bool,
        sumEntropyCheck: bool,
        entropyCheck: bool,
        differenceVarianceCheck: bool,
        differenceEntropyCheck: bool,
        informationMeasuresOfCorrelation12Check: bool,
        informationMeasuresOfCorrelation13Check: bool,
        sevenInvariantHuMomentsCheck: bool
    ):
        descriptors_flags = [
            energyCheck,
            contrastCheck,
            correlationCheck,
            varianceCheck,
            homogeneityCheck,
            sumAverageCheck,
            sumVarianceCheck,
            sumEntropyCheck,
            entropyCheck,
            differenceVarianceCheck,
            differenceEntropyCheck,
            informationMeasuresOfCorrelation12Check,
            informationMeasuresOfCorrelation13Check
        ]

        self.indexes_of_the_used_descriptors = [
            i for i in range(len(descriptors_flags)) if descriptors_flags[i]
        ]

        if sevenInvariantHuMomentsCheck:
            self.indexes_of_the_used_descriptors.extend(
                list(range(len(descriptors_flags), len(descriptors_flags) + 7))
            )

    def create_and_compile_model(self, num_neurons):
        input_shape = (len(self.indexes_of_the_used_descriptors),)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(num_neurons, activation=tf.nn.relu),
            # tf.keras.layers.Dense(256, activation=tf.nn.sigmoid),
            # tf.keras.layers.Dense(1024, activation=tf.nn.softmax),
            tf.keras.layers.Dense(4, activation=tf.nn.softmax),
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )

    def get_confusion_matrix(self) -> np.ndarray:
        predictions = self.model.predict(self.images_testing_set)
        predictions = np.argmax(predictions, axis=1)

        return tf.math.confusion_matrix(
            self.images_testing_labels_set,
            predictions
        ).numpy()

    @staticmethod
    def plot_confusion_matrix(
        confusion_matrix: np.ndarray,
        classes,
        normalize=False,
        title='Confusion matrix',
        color_map=plt.cm.Blues
    ):
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=color_map)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            confusion_matrix = (
                confusion_matrix.astype('float') /
                confusion_matrix.sum(axis=1)[:, np.newaxis]
            )

        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(
            range(confusion_matrix.shape[0]),
            range(confusion_matrix.shape[1]
        )):
            plt.text(j, i, confusion_matrix[i, j],
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @staticmethod
    def get_metrics(confusion_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
        FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
        TP = np.diag(confusion_matrix)
        TN = confusion_matrix.sum() - (FP + FN + TP)
        TPR = TP / (TP + FN)  # Sensitivity, hit rate, recall, or true positive rate
        TNR = TN / (TN + FP)  # Specificity or true negative rate
        ACC = (TP + TN) / (TP + FP + FN + TN)  # Overall accuracy
        sensitivity = round(TPR.mean(), 2)
        specificity = round(TNR.mean(), 2)
        accuracy = TP.sum() / confusion_matrix.sum()

        return {
            'FP': FP,
            'FN': FN,
            'TP': TP,
            'TN': TN,
            'TPR': TPR,
            'TNR': TNR,
            'ACC': ACC,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'accuracy': accuracy
        }

    def train(self, num_neurons = 256, num_epochs = 2000):
        self.create_and_compile_model(num_neurons)
        self.get_training_and_test_set()

        start = time.time()
        self.model.fit(
            self.images_training_set,
            self.images_training_labels_set,
            epochs=num_epochs
        )
        executionTime = time.time() - start

        confusion_matrix = self.get_confusion_matrix()
        self.plot_confusion_matrix(confusion_matrix, self.BIRADS_CLASSES)
        return executionTime, self.get_metrics(confusion_matrix)

    def train_different_number_of_neurons_and_epochs(self):
        num_neurons_arr = [32, 64, 128, 256, 512, 1024]
        num_epochs_arr = [100, 500, 1000, 2000, 3000, 4000, 5000]
        results: List[List[float]] = []

        for num_neurons in num_neurons_arr:
            for num_epochs in num_epochs_arr:
                execution_time, metrics = self.train(num_neurons, num_epochs)
                results.append([
                    execution_time,
                    metrics['accuracy'],
                    metrics['sensitivity'],
                    metrics['specificity'],
                    num_neurons,
                    num_epochs,
                ])

        with open('train_results.csv', 'w') as results_file:
            csv_writer = csv.writer(results_file)
            csv_writer.writerow([
                'Execution time (sec)',
                'Accuracy',
                'Sensitivity',
                'Specificity',
                'Num. of neurons',
                'Num. of epochs',
            ])
            csv_writer.writerows(results)

        return results

    def predict(self, image):
        predictions = self.model.predict(np.array([
            self.get_image_descriptors(image)
        ]))

        return np.argmax(predictions[0])

    @staticmethod
    def load_images_from_dir(dirname: str):
        images: List[Image.Image] = []

        # percorre os arquivos e diretórios que existirem dentro de imagens/1 por exemplo
        for dir_path, dirs, files in os.walk(os.path.join("imagens", dirname)):
            for file in files:
                if not "_cropped" in file and ".png" in file:
                    complete_path = os.path.join(dir_path, file)
                    # print('Loading image:', complete_path)
                    images.append(Image.open(complete_path))
        
        return images

    def load_images(self, show_msg_box = True):
        if not os.path.exists('imagens'):
            messagebox.showinfo('Error', 'The images folder was not found')
            exit(0)

        for birads_class in self.BIRADS_CLASSES:
            self.images[birads_class] = self.load_images_from_dir(birads_class)

        if show_msg_box: messagebox.showinfo('Concluded', 'Images were loaded')
