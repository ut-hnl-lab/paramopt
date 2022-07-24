import os.path as osp
from datetime import datetime
from typing import Any


def formatted_now() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def indent_repr(m: str, depth: int = 4, sep: str = "\n") -> str:
    blank = " " * depth
    return blank + m.replace(sep, "\n"+blank)


def unique_path(path):
    ret = path
    stem, ext = osp.splitext(path)
    i = 1
    while osp.isfile(ret):
        ret = stem + f' ({i})' + ext
        i += 1
    return ret
