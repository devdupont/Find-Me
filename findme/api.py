"""
Michael duPont - michael@mdupont.com

Usage: gunicorn -b 0.0.0.0 api:APP
"""

import falcon

class FindMeAPI:
    """"""

    def on_get(self, req: 'request', resp: 'response'):
        """"""
        resp.body = 'GET'

    def on_post(self, req: 'request', resp: 'response'):
        """"""
        resp.body = 'POST'

APP = falcon.API()
APP.add_route('/findme', FindMeAPI())
