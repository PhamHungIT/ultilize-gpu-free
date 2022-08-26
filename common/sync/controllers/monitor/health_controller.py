from flask import Response, request

from flask_restplus import Namespace, Resource

api = Namespace('health', description='health check')


@api.route('')
class Health(Resource):
    def get(self):
        return "Success", 200
