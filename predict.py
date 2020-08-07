import tensorflow as tf
import hparam as p
import cv2
import os

from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from magenta.models.sketch_rnn.utils import *

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_data():
    train_set = list()
    for i in range(1000):
        train_img = cv2.imread(p.train_path + str(i) + ".jpg")
        train_set.append(train_img)

    train_set = np.asarray(train_set, np.float32) / 127.5 - 1
    # train_set = rgb2bgr(train_set)
    sketches = np.load(p.label_path + "sketch.npy")
    return train_set, sketches


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


if __name__ == '__main__':
    t_x, s = get_data()
    IMG_SHAPE = (p.size, p.size, 3)
    VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    VGG16_MODEL.trainable = True
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(500, activation=None)
    model = tf.keras.Sequential([
        VGG16_MODEL,
        global_average_layer,
        prediction_layer
    ])

    if os.listdir(p.log_root):
        model.load_weights(p.log_root)

    for i in range(10):
        index = np.random.randint(20, 1000)
        x = np.reshape(t_x[index], [1, p.size, p.size, 3])
        skh = s[index]
        sketch_re = np.reshape(model.predict(x, 1), [250, 2])
        skh[:len(skh), :2] = sketch_re[:len(skh), :2]
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
        draw(skh, ax=ax)
        plt.show()




