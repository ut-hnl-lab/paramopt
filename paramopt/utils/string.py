import os.path as osp
from datetime import datetime


def formatted_now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _unique_path(path):
    ret = path
    stem, ext = osp.splitext(path)
    i = 1
    while osp.isfile(ret):
        ret = stem + f' ({i})' + ext
        i += 1
    return ret
