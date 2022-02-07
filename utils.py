import logging

LEVEL = logging.INFO


def get_logger(name: str):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(filename)s:%(lineno)d %(levelname)-8s %(message)s', '%y-%m-%d %H:%M:%S')
    # formatter = logging.Formatter('[%(asctime)s] %(filename)s:%(lineno)d %(levelname)s - %(message)s','%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(LEVEL)
    return logger
