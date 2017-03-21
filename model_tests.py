"""
Michael duPont - michael@mdupont.com
Testing for individual models
"""

#libraries
import begin
from cv2 import rectangle
from matplotlib.image import imread, imsave
#module
from findme.models import find_faces

def draw_boxes(bboxes: [[int]], img: 'np.array') -> 'np.array':
    """Returns an image array with the bounding boxes drawn around potential faces"""
    for x, y, w, h in bboxes:
        rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img

def test_find_faces():
    """Find faces for each test image"""
    base = 'test_imgs/find_faces/test{}.jpg'
    for i in range(1, 4):
        img = imread(base.format(i))
        bboxes = find_faces(img)
        print(bboxes)
        imsave(base.format(str(i)+'.out'), draw_boxes(bboxes, img))

@begin.start
def main(ff: 'Runs find_faces test'=False):
    """Test individual model components"""
    if ff:
        test_find_faces()
