from flask import Blueprint

from common.sync.controllers.monitor.health_controller import api as health_ns
from flask_restplus import Api

blueprint = Blueprint('api_monitor', __name__, url_prefix='/monitor')

api = Api(blueprint,
          title='monitor api',
          version='1.0',
          description='Author: hanv89'
          )
api.add_namespace(health_ns, path='/ping')
