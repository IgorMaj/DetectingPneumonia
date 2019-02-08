import os
import cv2
import numpy as np
from Constants import Constants


class ImageLoader:

    def __init__(self,root_path="./chest_xray"):

        self.paths = {}
        self._set_types = ["train", "val", "test"]
        for set_type in self._set_types:
            self.paths[set_type] = []

        for root, dirs, files in os.walk(root_path):
            for f in files:
                if f.endswith(".jpeg") or f.endswith(".jpg"):
                    self._add_to_right_set(os.path.join(root, f))


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
            #img_elem_gs = cv2.cvtColor(img_elem, cv2.COLOR_RGB2GRAY)
            img_resized = cv2.resize(img_elem_gs, (Constants.IMG_HEIGHT, Constants.IMG_WIDTH), interpolation=cv2.INTER_AREA)
            if normalize:
                #img_resized = img_resized/255.0
                img_resized = (img_resized - np.mean(img_resized))/np.std(img_resized)
            x.append(img_resized)
            y.append(self._extract_label(self.paths[set_type][i].lower()))

        return np.array(x), np.array(y)

    # 0 - normal , 1 - bacteria 2 - virus
    def _extract_label(self,str_path):
        if "normal" in str_path:
            return 0
        elif "bacteria" in str_path:
            return 1
        elif "virus" in str_path:
            return 2
        raise RuntimeError("Invalid image path: "+str_path)

