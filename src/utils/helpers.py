import atexit
import logging
import os


def setup_logger(run_id, log_dir='./logs'):
    os.makedirs(log_dir, exist_ok=True)
    log_fname = f"{log_dir}/{run_id}.log"
    logger = logging.getLogger()  # get root logger
    file_handler = logging.FileHandler(log_fname, mode="a", delay=False)
    file_handler.setFormatter(
        logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S'))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)  # all other loggers propagate to root; write to one log file from root
    print(f"Log path: {log_fname}")
    atexit.register(lambda: print(f"Log path: {log_fname}"))


class Dict2Obj(object):
    def __init__(self, dictionary):
        """Constructor"""
        for key in dictionary:
            setattr(self, key, dictionary[key])


def rename_key_in_dict(dictionary, oldkey, newkey):
    if type(dictionary) is dict:
        keys = list(dictionary)
        for k in keys:
            if k == oldkey:
                dictionary[newkey] = dictionary.pop(oldkey)
                rename_key_in_dict(dictionary[newkey], oldkey, newkey)
            else:
                rename_key_in_dict(dictionary[k], oldkey, newkey)
    elif type(dictionary) is list:
        for item in dictionary:
            rename_key_in_dict(item, oldkey, newkey)


def deep_get(obj, *keys, default={}):
    rtn = obj
    if type(rtn) is not dict:
        return default
    for k in keys:
        rtn = rtn.get(k, default)
        if type(rtn) is not dict:
            return rtn
    return rtn
