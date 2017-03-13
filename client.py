import requests

class FindMeClient:
    """"""

    url = 'localhost:8000/findme'

    def __init__(self, user: str):
        self.user = user

    def train(self, img: 'np.ndarray or path', value: bool):
        """"""
        if isinstance(img, str):
            img = open(img)
        requests.post(self.url, body=img, header={'user': self.user, 'value': value})

    def train_list(self, imgs: ['np.ndarray or path'], values: [bool]):
        """"""
        for img, value in zip(imgs, values):
            self.train(img, value)

    def evaluate(self, img: 'np.ndarray or path') -> bool:
        """"""
        if isinstance(img, str):
            img = open(img)
        resp = requests.get(self.url, body=img, header={'user': self.user})
        return bool(resp)
