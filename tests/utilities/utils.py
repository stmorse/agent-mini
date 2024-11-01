import hashlib          # uuid_to_int64
import configparser     # load_config
import os               # load_config
# import datetime
# import json
# import logging
# import random
# import re
# import sys

def uuid_to_int64(uuid_obj):
    '''
    Converts a UUID object into a FAISS ID compatible BIGINT
    '''
    uuid_string = str(uuid_obj)
    return int(hashlib.sha256(uuid_string.encode('utf-8')).hexdigest(), 16) % (2 ** 63)

def load_config(config_file):
    """
    Loads config from config_file
    :param config_file: path and filename for config.ini file
    :return: configparser.ConfigParser object or None if the file does not exist
    """

    if not os.path.exists(config_file):
        print(f"Config file {config_file} does not exist.")
        return None

    config = configparser.ConfigParser()
    config.read(config_file)

    # Checking if the config file was empty or improperly formatted
    if not config.sections():
        print(f"Config file {config_file} is empty or improperly formatted.")
        return None

    return config

