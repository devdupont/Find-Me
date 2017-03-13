"""
Michael duPont - michael@mdupont.com
"""

import cv2
import requests

class FindMeClient:
    """"""

    url = 'localhost:8000/findme'

    def __init__(self, user: str):
        self.user = user

    def upload(self, img: 'np.ndarray or path', value: bool, debug: bool=False):
        """Upload an image to a user's image corpus.
        The image must contain just a single face
        """
        if isinstance(img, str):
            img = cv2.imread(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resp = requests.put(self.url, body=img, header={'user': self.user, 'value': value})
        if debug:
            x, y, w, h = resp.body['face']
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite()

    def upload_list(self, imgs: ['np.ndarray or path'], values: [bool]):
        """Upload multiple images to a user's image corpus"""
        for img, value in zip(imgs, values):
            self.upload(img, value)

    def train(self):
        """"""
        requests.post(self.url, header={'user': self.user})

    def evaluate(self, img: 'np.ndarray or path') -> bool:
        """"""
        if isinstance(img, str):
            img = open(img)
        resp = requests.get(self.url, body=img, header={'user': self.user})
        return bool(resp)
