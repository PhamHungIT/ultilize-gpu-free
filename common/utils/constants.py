ROLE_OWNER = "owner"
ROLE_MAINTAINER = "maintainer"
ROLE_SUPPORTER = "supporter"
ROLE_REPORTER = "reporter"
ROLE_VIEWER = "viewer"
role_priority = {'owner': 99,
                 'maintainer': 98,
                 'supporter': 97,
                 'reporter': 96,
                 'viewer': 95}

STATUS_CREATED = 'created'
STATUS_ACTIVATED = 'activated'
STATUS_APPROVED = 'approved'
STATUS_BANNED = 'banned'
status_priority = {'approved': 99,
                   'activated': 98,
                   'created': 97,
                   'banned': 96}

AGENT_STATUS_DISABLED = 'disabled'
AGENT_STATUS_ENABLED = 'enabled'

DEFAULT_VERSION = 'v1'
DEFAULT_ENVIRONMENT = 'Development'
DEFAULT_MODEL_NAME = 'default'
DEFAULT_MODEL_VERSION = 'v1'
DEFAULT_CONFIDENCE_START = 40
DEFAULT_CONFIDENCE_END = 70

TRAINING_STATUS_PREPARING = 'PREPARING'
TRAINING_STATUS_TRAINING = 'TRAINING'
TRAINING_STATUS_STOPPING = 'STOPPING'
TRAINING_STATUS_STOPPED = 'STOPPED'
TRAINING_STATUS_DONE = 'DONE'
