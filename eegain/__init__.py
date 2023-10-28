import logging
import os
import sys

from eegain import transforms
from eegain.models import list_models

from .version import __version__

# implement logging system
file_handler = logging.FileHandler("log.log", mode="w")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

stream_handler = logging.StreamHandler(stream=sys.stdout)
stream_handler.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
stream_handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])

logger = logging.getLogger("EEGain")
