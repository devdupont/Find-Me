"""
Michael duPont - michael@mdupont.com
"""

#stdlib
import pickle
from os import path, mkdir
#libraries
import cv2
from keras.layers import Activation, Convolution2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from keras.models import Sequential, Model, model_from_json
from sklearn.utils import shuffle

##---------- Face Finder ----------##

CASCADE = cv2.CascadeClassifier(__file__.replace('models.py', '')+'haar_cc_front_face.xml')

def find_faces(img: 'np.ndarray') -> [(int)]:
    """Returns a list of bounding boxes for every face found in an image"""
    return CASCADE.detectMultiScale(
        cv2.cvtColor(img, cv2.COLOR_RGB2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

##---------- Facial Recognition Model ----------##

#TODO: test loading single model.json file

def make_model() -> Sequential:
    """Create a Sequential Keras model to boolean classify faces"""
    model = Sequential()
    return model

def get_model(user: str) -> Sequential:
    """Load a user's existing model or create a new one"""
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

def add_img(user, img):
    """Add an image to a user's image corpus. Creates one is none exists"""
    fpath = 'user_models/'+user
    if not path.isdir(fpath):
        mkdir(fpath)
    fpath += '/imgs.pkl'
    corpus = pickle.load(open(fpath, 'rb')) if path.isfile(fpath) else []
    corpus.append(img)
    pickle.dump(corpus, open(fpath, 'wb'))

def load_data(user: str) -> (['np.ndarray'], [bool]):
    """Returns a shuffled list of base and user feature-value pairs for training"""
    fpath = 'user_models/'+user
    if not path.isdir(fpath):
        return None
    corpus = pickle.load(open('base_corpus.pkl', 'rb'))
    values = [False] * len(corpus)
    ucorpus = pickle.load(open(fpath+'/imgs.pkl', 'rb'))
    corpus += ucorpus
    values += [True] * len(ucorpus)
    return shuffle(corpus, values)

def train(user: str):
    """Train a user's model over their uploaded corpus"""
    imgs, values = load_data(user)
    model = get_model(user)
    model.fit(imgs, values, samples_per_epoch=len(values), nb_epoch=5)
    save_model(user, model)

def evaluate(user, img) -> bool:
    """Determine if a face is a user's target according to their trained model"""
    model = get_model(user)
    ret = model.predict(img, batch_size=1)
    return ret
