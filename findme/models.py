"""
Michael duPont - michael@mdupont.com
Models - Contains the ML models used for facial recognition
"""

#stdlib
# from pickle import load
# from os import path
#library
import cv2
import numpy as np
# from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
# from keras.models import Sequential
# from keras.wrappers.scikit_learn import KerasClassifier
# from sklearn.utils import shuffle

LOC = __file__.replace('models.py', '')

CASCADE = cv2.CascadeClassifier(LOC+'haar_cc_front_face.xml')

def find_faces(img: np.ndarray, sf=1.16, mn=5) -> np.array([[int]]):
    """Returns a list of bounding boxes for every face found in an image"""
    return CASCADE.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
        scaleFactor=sf,
        minNeighbors=mn,
        minSize=(45, 45),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

# SHAPE = None

# def make_model() -> Sequential:
#     """Create a Sequential Keras model to boolean classify faces"""
#     model = Sequential()
#     # First Convolution
#     model.add(Convolution2D(32, (5, 5), input_shape=SHAPE))
#     model.add(Activation('relu'))
#     model.add(Dropout(.2))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     # Second Convolution
#     model.add(Convolution2D(32, (5, 5)))
#     model.add(Activation('relu'))
#     model.add(Dropout(.2))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#     # Flatten for Fully Connected
#     model.add(Flatten())
#     # First Fully Connected
#     model.add(Dense(128))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     # Second Fully Connected
#     model.add(Dense(64))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.2))
#     # Output
#     model.add(Dense(1)) #2
#     model.add(Activation('softmax'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     #print(model.summary())
#     return model

# def train_model(X: 'Array of image arrays', Y: [int]) -> 'trained Sequential':
#     """"""
#     global SHAPE
#     SHAPE = X[0].shape
#     model = KerasClassifier(build_fn=make_model, epochs=5, batch_size=len(Y), verbose=0)
#     model.fit(*shuffle(X, Y, random_state=42))
#     return model
# 
# def get_model():
#     """Returns a saved model or trains a new one"""
#     if path.isfile(LOC+'model.json') and path.isfile(LOC+'model.h5'):
#         model = model_from_json(open(LOC+'model.json').read())
#         model.compile('adam', 'mse')
#         model.load_weights(LOC+'model.h5')
#         return model
#     else:
#         X, Y = load(open(LOC+'features.pkl', 'rb')), load(open(LOC+'labels.pkl', 'rb'))
#         model = train_model(X, Y)
#         print(model.to_json(), file=open(LOC+'model.json', 'w'))
#         model.save_weights(LOC+'model.h5')
#         return model
