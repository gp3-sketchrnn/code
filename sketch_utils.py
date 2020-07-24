import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
from matplotlib.path import Path
from magenta.models.sketch_rnn.utils import *
import cv2
import tensorflow.compat.v1 as tf


class SketchPath(Path):

    def __init__(self, data, factor=.2, *args, **kwargs):
        vertices = np.cumsum(data[::, :-1], axis=0) / factor
        codes = np.roll(self.to_code(data[::, -1].astype(int)),
                        shift=1)
        codes[0] = Path.MOVETO

        super(SketchPath, self).__init__(vertices, codes, *args, **kwargs)

    @staticmethod
    def to_code(cmd):
        # if cmd == 0, the code is LINETO
        # if cmd == 1, the code is MOVETO (which is LINETO - 1)
        return Path.LINETO - cmd


def draw(sketch_data, factor=.2, pad=(10, 10), ax=None):
    if ax is None:
        ax = plt.gca()

    x_pad, y_pad = pad

    x_pad //= 2
    y_pad //= 2

    x_min, x_max, y_min, y_max = get_bounds(data=sketch_data, factor=factor)

    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_max + y_pad, y_min - y_pad)

    sketch = SketchPath(sketch_data)

    patch = patches.PathPatch(sketch, facecolor='none')
    ax.add_patch(patch)


def sketch_2_img(strokes):
    sketch_reconstructed = to_normal_strokes(strokes)
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    draw(sketch_reconstructed, ax=ax)
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.resize(data, (128, 128))
    return np.reshape(img, (1, 128, 128, 3))


def get_suggested_point(sess, cnn_model, img):
    images = tf.placeholder("float", [1, 128, 128, 3])
    feed_dict = {images: img}
    prob = sess.run(cnn_model.prob, feed_dict=feed_dict)
    return prob[0], prob[1]
