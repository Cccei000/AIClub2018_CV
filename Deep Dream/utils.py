import PIL.Image
from io import BytesIO
from IPython.display import Image, display
import numpy as np


def show_and_save(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    img = PIL.Image.fromarray(a)
    f = BytesIO()
    img.save(f,fmt)
    img.save('img.jpg')
    display(Image(data=f.getvalue()))


def showImg(a):
    mean = np.array([0.485,0.456,0.406]).reshape([1,1,3])
    std = np.array([0.229,0.224,0.225]).reshape([1,1,3])
    inp = a
    inp = inp.transpose(1,2,0)
    inp = std*inp+mean
    inp *= 255
    show_and_save(inp)