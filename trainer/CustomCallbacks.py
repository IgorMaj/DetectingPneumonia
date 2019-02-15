from keras.callbacks import Callback
import pickle
from trainer.Constants import Constants
from tensorflow.python.lib.io import file_io


class ModelSaveCallback(Callback):

    def __init__(self, runs_on_cloud=False):
        self.runs_on_cloud = runs_on_cloud

    def on_epoch_end(self, epoch, logs=None):
        if not self.runs_on_cloud:
            pickle.dump(self.model, open(Constants.MODEL_FILE_PATH, mode="wb"))
        else:
            pickle.dump(self.model, file_io.FileIO(Constants.CLOUD_BUCKET + "/" + Constants.MODEL_FILE_PATH, mode='wb'))