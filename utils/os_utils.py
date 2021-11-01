import os
import subprocess
import shlex
import logging

def get_dir():
    utils_dirname = os.path.dirname(os.path.abspath(__file__))
    lib_dirname = os.path.abspath(os.path.join(utils_dirname, ".."))
    return lib_dirname

def shell(cmd):
    logging.info("running shell command: \n {}".format(cmd))
    subprocess.check_call(shlex.split(cmd))

def mkdir_p(dir_path):
    shell("mkdir -p {}".format(dir_path))

def mkdir_p_local(relative_dir_path):
    """create folder inside of library if does not exists"""
    local_dir = get_dir()
    abspath = os.path.join(local_dir, relative_dir_path)
    mkdir_p(abspath)

def console_logging():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
