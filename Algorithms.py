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
