"""
Michael duPont - michael@mdupont.com
FindMe - Primary facial recognition and location function
"""

import numpy as np
from findme.imageutil import crop, preprocess
from findme.models import find_faces#, get_model

# MODEL = get_model()

# def target_in_img(img: np.ndarray) -> (bool, np.array([int])):
#     """Returns whether the target is in a given image and where"""
#     for bbox in find_faces(img):
#         face = preprocess(crop(img, *bbox))
#         if MODEL.predict(np.array([face])) == 1:
#             return True, bbox
#     return False, None
