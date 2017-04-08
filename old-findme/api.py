"""
Michael duPont - michael@mdupont.com

Usage: gunicorn -b 0.0.0.0 api:APP
"""

#stdlib
from json import dumps
#libraries
import falcon
#module
from findme.models import find_faces, add_img, train, evaluate

def crop(img: 'np.ndarray', bounding_box: [int]) -> 'np.ndarray':
    """Crop an image to a bounding box (x,y,w,h)"""
    x, y, w, h = bounding_box
    return img[y: y+h, x: x+w]

class FindMeAPI:
    """"""

    def verify_req(self, req: 'request', headers: [str]) -> bool:
        """"""
        return True

    def error(self, msg: str, **kwargs) -> str:
        """"""
        ret = {'Error': msg}
        ret.update(kwargs)
        return dumps(ret)

    def on_get(self, req: 'request', resp: 'response'):
        """Evaluate an image to determine if a user's target face is found"""
        if not self.verify_req(req, ['user']):
            resp.body = 'Not a valid request'
            return
        img = req.body
        faces = find_faces(img)
        if not faces:
            resp.body = self.error('No faces found')
            return
        user = req.headers['user']
        for bbox in faces: #(x, y, w, h)
            face = crop(img, bbox)
            if evaluate(user, face):
                resp.body = dumps({'success': True, 'face': bbox})
                return
        resp.body = dumps({'success': False})

    def on_post(self, req: 'request', resp: 'response'):
        """Train a user's model on the uploaded image corpus"""
        if not self.verify_req(req, ['user', 'value']):
            resp.body = self.error('Not a valid request')
            return
        user = req.headers['user']
        train(user)
        resp.body = dumps({'Success': True})

    def on_put(self, req: 'request', resp: 'response'):
        """Add a new image (cropped face) to the user's training corpus"""
        if not self.verify_req(req, ['user']):
            resp.body = 'Not a valid request'
            return
        img = req.body
        faces = find_faces(img)
        if not faces:
            resp.body = self.error('No face found')
            return
        elif len(faces) > 1:
            resp.body = self.error('More than one face found', Faces=faces)
        user = req.headers['user']
        face = crop(img, faces[0])
        add_img(user, face)
        resp.body = dumps({'face': faces[0]})

APP = falcon.API()
APP.add_route('/findme', FindMeAPI())
