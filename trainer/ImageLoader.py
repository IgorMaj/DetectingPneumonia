import os
import cv2
import numpy as np
from trainer.Constants import Constants
from abc import ABC, abstractmethod
#import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from io import BytesIO
import random




"""def show_image(image, color="gray"):
    plt.imshow(image, color)
    plt.show()"""


class ImageLoader(ABC):
    @abstractmethod
    def load_set(self, set_type, num_elems=None, normalize=False):
        pass

# 0 - normal , 1 - bacteria 2 - virus
    def _extract_label(self,str_path):
        if "normal" in str_path:
            return 0
        elif "bacteria" in str_path:
            return 1
        elif "virus" in str_path:
            return 2
        raise RuntimeError("Invalid image path: "+str_path)


class FileImageLoader(ImageLoader):

    def __init__(self,root_path="./chest_xray"):

        self.paths = {}
        self._set_types = ["train", "val", "test"]
        for set_type in self._set_types:
            self.paths[set_type] = []

        for root, dirs, files in os.walk(root_path):
            for f in files:
                if f.endswith(".jpeg") or f.endswith(".jpg"):
                    self._add_to_right_set(os.path.join(root, f))

        for set_type in self._set_types:
            random.shuffle(self.paths[set_type])

    def _add_to_right_set(self, file_path):

        str_path = str(file_path).lower()
        for set_type in self._set_types:
            if set_type in str_path:
                self.paths[set_type].append(file_path)

    def load_set(self, set_type, num_elems=None,normalize=False):
        x = []
        y = []
        if num_elems is None:
            num_elems = len(self.paths[set_type])

        for i in range(num_elems):
            # konverzija u grayscale
            img_elem_gs = cv2.imread(self.paths[set_type][i], cv2.IMREAD_GRAYSCALE)
            img_resized = cv2.resize(img_elem_gs, (Constants.IMG_HEIGHT, Constants.IMG_WIDTH), interpolation=cv2.INTER_AREA)
            if normalize:
                img_resized = (img_resized - np.mean(img_resized))/np.std(img_resized)
            x.append(img_resized)
            y.append(self._extract_label(self.paths[set_type][i].lower()))

        return np.array(x), np.array(y)


class CloudImageLoader(ImageLoader):
    def __init__(self, job_dir_path=""):
        self.root_path = job_dir_path+"/chest_xray/"
        self.bucket_name = job_dir_path

    def create_opencv_image_from_io(self,img_stream, cv2_img_flag=0):
        img_stream.seek(0)
        img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
        return cv2.imdecode(img_array, cv2_img_flag)

    def process_element(self, path, x, y, normalize):
        for elem in tf.gfile.ListDirectory(path):
            new_path = path+"/"+elem
            if tf.gfile.IsDirectory(new_path):
                self.process_element(new_path, x, y,normalize)
            elif new_path.endswith(".jpeg") or new_path.endswith(".jpg"):
                new_path = new_path.replace("//", "/").replace("/", "//", 1)
                #print(new_path)
                file = BytesIO(file_io.read_file_to_string(new_path,binary_mode=True))
                img_elem_gs = self.create_opencv_image_from_io(file,cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img_elem_gs, (Constants.IMG_HEIGHT, Constants.IMG_WIDTH),
                                         interpolation=cv2.INTER_AREA)
                if normalize:
                    img_resized = (img_resized - np.mean(img_resized)) / np.std(img_resized)
                x.append(img_resized)
                y.append(self._extract_label(new_path.lower()))

    def load_set(self, set_type, num_elems=None, normalize=False):
        x = []
        y = []
        self.process_element(self.root_path+set_type, x, y, normalize)
        return np.array(x),np.array(y)