from os import path, mkdir
import cv2
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential, Model, model_from_json

##---------- Face Finder ----------##

CC_WEIGHTS = ''
CASCADE = cv2.CascadeClassifier(CC_WEIGHTS)

def find_faces(img: 'np.ndarray') -> [(int)]:
    """"""
    return CASCADE.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

##---------- Facial Recognition Model ----------##

#TODO: test loading single model.json file

def make_model() -> Sequential:
    """"""
    model = Sequential()
    return model

def get_model(user: str) -> Sequential:
    """"""
    fpath = 'user_models/'+user
    if path.isdir(fpath):
        model = model_from_json(open(fpath+'/model.json').read())
        model.compile("adam", "mse")
        model.load_weights(fpath+'/model.h5')
        return model
    else:
        return make_model()

def save_model(user: str, model: Sequential):
    """Saves a model as json and h5 files in a user's model dir"""
    fpath = 'user_models/'+user
    if not path.isdir(fpath):
        mkdir(fpath)
    print(model.to_json(), file=open(fpath+'/model.json', 'w'))
    model.save_weights(fpath+'/model.h5')

def train(user, img):
    """"""
    model = get_model(user)
    #do stuff
    save_model(user, model)

def evaluate(user, img) -> bool:
    """"""
    model = get_model(user)
    #do stuff
    return True
