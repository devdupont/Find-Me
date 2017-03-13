"""
Michael duPont - michael@mdupont.com

Usage: gunicorn -b 0.0.0.0 api:APP
"""

from json import dumps
import falcon
from cv2 import crop
from models import find_faces, train, evaluate

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
        """"""
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
        """"""
        if not self.verify_req(req, ['user', 'value']):
            resp.body = self.error('Not a valid request')
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
        train(user, face)
        resp.body = dumps({'face': faces[0]})

APP = falcon.API()
APP.add_route('/findme', FindMeAPI())
