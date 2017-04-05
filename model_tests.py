"""
Michael duPont - michael@mdupont.com
Testing for individual models
"""

#stdlib
from glob import glob
from pickle import dump
#libraries
import begin
from cv2 import rectangle
from matplotlib.image import imread, imsave
#module
from findme.models import find_faces, train, evaluate

def draw_boxes(bboxes: [[int]], img: 'np.array') -> 'np.array':
    """Returns an image array with the bounding boxes drawn around potential faces"""
    for x, y, w, h in bboxes:
        rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

def test_find_faces():
    """Find faces for each test image"""
    for fname in glob('test_imgs/group*.jpg'):
        img = imread(fname)
        bboxes = find_faces(img)
        print(bboxes)
        imsave(fname.replace('/', '/find_faces/'), draw_boxes(bboxes, img))

##----------  ----------##

def corpus_create():
    """Creates cropped faces for imgs matching 'test_imgs/group*.jpg'"""
    i = 0
    base = 'test_imgs/corpus/face{}.jpg'
    for fname in glob('test_imgs/group*.jpg'):
        img = imread(fname)
        bboxes = find_faces(img)
        for x, y, w, h in bboxes:
            cropped = img[y:y+h, x:x+w]
            imsave(base.format(i), cropped)
            i += 1

def corpus_update():
    """Creates base_corpus.pkl from face imgs in test_imgs/corpus"""
    imgs = [imread(fname) for fname in glob('test_imgs/corpus/face*.jpg')]
    dump(imgs, open('findme/base_corpus.pkl', 'wb'))

##----------  ----------##

def test_model_train():
    train('mdupont')

def test_model_evaluate():
    base = 'test_imgs/evaluate/test{}.jpg'
    for i in range(1, 4):
        img = imread(base.format(i))
        evaluate('mdupont', img)

##----------  ----------##

@begin.start
def main(ff: 'Runs find_faces test'=False,
         cc: 'Create face corpus'=False,
         cu: 'Update corpus pkl'=False,
         mt: 'Trains model'=False,
         me: 'Tests facial recognition evaluations'=False):
    """Test individual model components"""
    if ff: test_find_faces()
    elif cc: corpus_create()
    elif cu: corpus_update()
    elif mt: test_model_train()
    elif me: test_model_evaluate()
