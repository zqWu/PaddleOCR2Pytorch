import logging
import sys

logging.basicConfig(
    level=logging.ERROR,
    stream=sys.stdout,
    format='[%(asctime)s] %(name)s %(levelname)s: %(message)s',
    datefmt="%Y/%m/%d %H:%M:%S"
)


def get_logger(name):
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    return log

# from logger import get_logger
# logger = get_logger(__name__)
