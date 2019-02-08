from ImageLoader import ImageLoader
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from Constants import Constants
import time
from sklearn.model_selection import train_test_split
import numpy as np


def create_model():
    print("Creating model...")
    model = Sequential()

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation="relu",
                            input_shape=(Constants.IMG_HEIGHT, Constants.IMG_WIDTH, 1)))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(Convolution2D(filters=128, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=12, activation="relu"))
    model.add(Dense(units=12, activation="relu"))
    model.add(Dense(units=3, activation="softmax"))

    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001), metrics=["accuracy"])
    return model


def show_image(image, color="gray"):
    plt.imshow(image, color)
    plt.show()


#Podesavanje modela (bazirano na vgg16)
model = create_model()

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

#start_time = time.time()
#print("Loading test data...")
#X_test, y_test = img_loader.load_set("test",normalize=True)
#print('Test data loaded in: ',time.time()-start_time)
#X_test = X_test.reshape(X_test.shape[0], Constants.IMG_HEIGHT, Constants.IMG_WIDTH, 1)
#y_test = np_utils.to_categorical(y_test, Constants.NUM_CATEGORIES)


print("Training model...")
model.fit(x=X_train, y=y_train, verbose=1, epochs=5)

print("Evaluating...")
loss, acc = model.evaluate(x=X_validate,y=y_validate,verbose=0)
print('Loss: ', loss)
print('Accuracy: ', acc)