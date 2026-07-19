import logging

from core.config.settings import get_settings
from core.logging.json_formatter import JSONFormatter

settings = get_settings()


logger = logging.getLogger('chatagent')
logger.setLevel(settings.log_level)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
