import os.path as osp
from typing import List

from natsort import natsorted
from PIL import Image


def unique_path(path):
    ret = path
    stem, ext = osp.splitext(path)
    i = 1
    while osp.isfile(ret):
        ret = stem + f' ({i})' + ext
        i += 1
    return ret


def create_gif(
    img_paths: List[str],
    duration: float = 1.0,
    loop: int = 0
) -> None:
    """Generate gif video from images.

    Parameters
    ----------
    img_paths : list of strs
        List of image paths
    duration : float, optional
        Time interval between frames [s], by default 1.0.
    loop : int, optional
        Number of loop, by default 0 (infinite)
    """
    if len(img_paths) == 0:
        print('No file')
        return
    folder = osp.dirname(img_paths[0])
    save_path = unique_path(osp.join(folder, 'animation.gif'))
    images = list(map(lambda file: Image.open(file), natsorted(img_paths)))
    images[0].save(
        save_path, save_all=True, append_images=images[1:],
        duration=float(duration)*1000, loop=int(loop))
    print('Done')


def select_images() -> List[str]:
    """Select image files using GUI dialog.

    Returns
    -------
    list of str
        List of image paths
    """
    from tkinter import Tk, filedialog
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(True)
    except:
        pass
    Tk().withdraw()
    img_paths = filedialog.askopenfilenames(
        filetypes=[
            ('PNG files', '*.png'), ('JPEG files', '*.jpg')], initialdir='.')
    return img_paths
