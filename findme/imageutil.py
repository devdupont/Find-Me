"""
Michael duPont - michael@mdupont.com
Image Utils - Functions for manipulating image arrays
"""

import cv2

def crop(img: 'np.ndarray', x: int, y: int, width: int, height: int) -> 'np.ndarray':
    """Returns an image cropped to a given bounding box of top-left coords, width, and height"""
    return img[y:y+height, x:x+width]

def preprocess(img: 'np.ndarray') -> 'np.ndarray':
    """Resizes a given image and remove alpha channel"""
    return cv2.resize(img, (45, 45), interpolation=cv2.INTER_AREA)[:,:,:3]

def draw_boxes(bboxes: [[int]], img: 'np.array', line_width: int=2) -> 'np.array':
    """Returns an image array with the bounding boxes drawn around potential faces"""
    for x, y, w, h in bboxes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), line_width)
    return img
