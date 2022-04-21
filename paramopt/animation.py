import os.path as osp
from typing import List
from PIL import Image

from natsort import natsorted


def create_gif(
    file_paths: List[str], duration: float = 1.0, loop: int = 0
) -> None:
    """Generate gif video from images.

    Parameters
    ----------
    file_paths : List[str]
        List of image paths
    duration : float, optional
        Time interval between frames [s], by default 1.0.
    loop : int, optional
        Number of loop, by default 0 (infinite)
    """
    if len(file_paths) == 0:
        print('No file')
        return
    folder = osp.dirname(file_paths[0])
    save_path = _unique_path(osp.join(folder, 'animation.gif'))
    images = list(map(lambda file: Image.open(file), natsorted(file_paths)))
    images[0].save(
        save_path, save_all=True, append_images=images[1:],
        duration=float(duration)*1000, loop=int(loop))
    print('Done')


def select_images() -> List[str]:
    """Select image files using GUI dialog.

    Returns
    -------
    List[str]
        List of image paths
    """
    from tkinter import filedialog, Tk
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(True)
    except:
        pass
    Tk().withdraw()
    file_paths = filedialog.askopenfilenames(
        filetypes=[
            ('PNGファイル', '*.png'), ('JPEGファイル', '*.jpg')], initialdir='.')
    return file_paths


def _unique_path(path):
    ret = path
    stem, ext = osp.splitext(path)
    i = 1
    while osp.isfile(ret):
        ret = stem + f' ({i})' + ext
        i += 1
    return ret
