import os
import random
import numpy as np
import tensorflow as tf
import mahotas
import cv2
import itertools
import time

from typing import List, Dict

from tkinter import messagebox

from PIL import Image

import matplotlib.pyplot as plt


class Algorithms:
    BIRADS_CLASSES = ["1", "2", "3", "4"]

    def __init__(self) -> None:
        self.images: Dict[str, List[Image.Image]] = {}

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

    @staticmethod
    def resample_image(image):
        return np.round(np.array(image) / 8).astype(np.uint8)

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
