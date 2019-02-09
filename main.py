from ImageLoader import ImageLoader
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from Constants import Constants
import time
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import load_model
import sys
from sklearn.metrics import confusion_matrix
import itertools



def create_or_load_model(load_model_from_file=False):
    if load_model_from_file:
        print("Loading model...")
        model = load_model(Constants.MODEL_FILE_PATH)
        return model

    print("Creating model...")
    model = Sequential()

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation="relu",
                            input_shape=(Constants.IMG_HEIGHT, Constants.IMG_WIDTH, 1), use_bias=False))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation="relu",use_bias=False))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation="relu", use_bias=False))
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation="relu", use_bias=False))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=4, activation="relu", use_bias=False))
    model.add(Dense(units=36, activation="relu", use_bias=False))
    model.add(Dense(units=Constants.NUM_CATEGORIES, activation="softmax", use_bias=False))

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    return model


def show_image(image, color="gray"):
    plt.imshow(image, color)
    plt.show()

#Argumenti su da li da ucitava model i koliko epoha da ga trenira
def get_args_from_cmd():
    if len(sys.argv) != 3:
        return False, 5
    return sys.argv[1] == 'True', int(sys.argv[2])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def convert_to_num(y):
    retval = []
    for element in y:
        for i in range(len(element)):
            if element[i] != 0:
                retval.append(i)
                break
    return np.array(retval)


if __name__ == '__main__':
    load_model_from_file, num_epochs = get_args_from_cmd()
    #Podesavanje modela (bazirano na vgg16)
    model = create_or_load_model(load_model_from_file)

    img_loader = ImageLoader()
    start_time = time.time()
    print("Loading training data...")
    X_train, y_train = img_loader.load_set("train",normalize=True)
    print('Training data loaded in: ',time.time()-start_time)
    # Preprocesiranje, da radi sa tensorflowom, pre je bilo (X_train.shape[0], 224, 224)
    X_train = X_train.reshape(X_train.shape[0], Constants.IMG_HEIGHT, Constants.IMG_WIDTH, 1)
    # Preprocesiranje pretvara (0-normal,1-bacteria,2-virus) u [0.,0.,1.] [0.,1.,0.] i [1.,0.,0.]
    y_train = np_utils.to_categorical(y_train, Constants.NUM_CATEGORIES)

    print("Spliting training data to train and validation sets and adding small validation set..")

    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2, random_state=None)
    # validacioni skup, nazvan small jer ima svega nekoliko slika
    X_small,y_small = img_loader.load_set("val", normalize=True)
    X_small = X_small.reshape(X_small.shape[0], Constants.IMG_HEIGHT, Constants.IMG_WIDTH, 1)
    y_small = np_utils.to_categorical(y_small, Constants.NUM_CATEGORIES)
    print(X_validate.shape)
    print(X_small.shape)
    X_validate = np.concatenate((X_validate, X_small), axis=0)
    y_validate = np.concatenate((y_validate,y_small), axis=0)
    print(X_validate.shape)
    print("Done!")

    """start_time = time.time()
    print("Loading test data...")
    X_test, y_test = img_loader.load_set("test",normalize=True)
    print('Test data loaded in: ',time.time()-start_time)
    X_test = X_test.reshape(X_test.shape[0], Constants.IMG_HEIGHT, Constants.IMG_WIDTH, 1)
    y_test = np_utils.to_categorical(y_test, Constants.NUM_CATEGORIES)"""

    print("Training model...")
    model.fit(x=X_train, y=y_train, verbose=1, epochs=num_epochs)
    print("Saving model...")
    model.save(Constants.MODEL_FILE_PATH)
    print("Evaluating...")
    loss, acc = model.evaluate(x=X_validate,y=y_validate,verbose=0)
    rounded_predictions = model.predict_classes(X_validate)
    cm = confusion_matrix(y_true=convert_to_num(y_validate), y_pred=rounded_predictions)
    plot_confusion_matrix(cm, Constants.CLASS_NAMES)
    print('Loss: ', loss)
    print('Accuracy: ', acc)
