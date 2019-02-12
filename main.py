from ImageLoader import ImageLoader
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from Constants import Constants
import time
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import load_model
import sys
from sklearn.metrics import confusion_matrix
import itertools
from keras.preprocessing.image import ImageDataGenerator


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
    #model.add(BatchNormalization())
    model.add(Dropout(rate=0.25))
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation="relu", use_bias=False))
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation="relu", use_bias=False))
    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(units=36, activation="relu", use_bias=False))
    model.add(Dense(units=36, activation="relu", use_bias=False))
    model.add(Dense(units=Constants.NUM_CATEGORIES, activation="softmax", use_bias=False))

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


# Argumenti su da li da ucitava model, koliko epoha da ga trenira,da li da pokrene test
# Primer: False(ne ucitava model), 5(broj epoha, 0 nece ucitavati train i validaciju), False(ne pokrece test)
def get_args_from_cmd():
    if len(sys.argv) != 4:
        return False, 5, False
    return sys.argv[1] == 'True', int(sys.argv[2]), sys.argv[3] == 'True'


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


def load_and_reshape_data(img_loader, chosen_set):
    start_time = time.time()
    print("Loading "+str(chosen_set) + " data...")
    x, y = img_loader.load_set(chosen_set,normalize=True)
    print(str(chosen_set)+' data loaded in: ',time.time()-start_time)
    x = x.reshape(x.shape[0], Constants.IMG_HEIGHT, Constants.IMG_WIDTH, 1)
    y = np_utils.to_categorical(y, Constants.NUM_CATEGORIES)
    return x, y


def evaluate_model(model,x,y):
    print("Evaluating...")
    loss, acc = model.evaluate(x=x, y=y, verbose=0)
    rounded_predictions = model.predict_classes(x)
    cm = confusion_matrix(y_true=convert_to_num(y), y_pred=rounded_predictions)
    plot_confusion_matrix(cm, Constants.CLASS_NAMES)
    print('Loss: ', loss)
    print('Accuracy: ', acc)


def train_and_evaluate_model(model, img_loader,num_epochs):
    x_train, y_train = load_and_reshape_data(img_loader, "train")
    print("Spliting training data to train and validation sets and adding small validation set..")
    x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=None)
    # validacioni skup, nazvan small jer ima svega nekoliko slika
    x_small, y_small = load_and_reshape_data(img_loader, "val")
    x_validate = np.concatenate((x_validate, x_small), axis=0)
    y_validate = np.concatenate((y_validate, y_small), axis=0)
    print("Done!")

    data_gen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=0.8,
        zoom_range=0.2
    )

    print("Training model...")
    model.fit_generator(data_gen.flow(x=x_train, y=y_train, batch_size=32), steps_per_epoch=len(x_train) / 32
                        , verbose=1, epochs=num_epochs)
    print("Saving model...")
    model.save(Constants.MODEL_FILE_PATH)
    evaluate_model(model, x_validate, y_validate)


if __name__ == '__main__':
    load_model_from_file, num_epochs,run_test = get_args_from_cmd()
    model = create_or_load_model(load_model_from_file)
    img_loader = ImageLoader()
    if num_epochs > 0:
        train_and_evaluate_model(model, img_loader, num_epochs)

    if run_test:
        print("Running test...")
        X_test, y_test = load_and_reshape_data(img_loader, "test")
        evaluate_model(model, X_test, y_test)