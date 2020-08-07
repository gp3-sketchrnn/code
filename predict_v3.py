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
    stk = np.random.random([10, 3]) * 20 - 10
    stk[:, 2] = 0

    IMG_SHAPE = (p.size, p.size, 3)
    main_input = tf.keras.Input((p.size, p.size, 3), name="ipt")
    VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    VGG16_MODEL.trainable = False
    vgg_result = VGG16_MODEL(main_input)
    fc = tf.keras.layers.Flatten()(vgg_result)
    dense = tf.keras.layers.Dense(1000, activation=tf.nn.elu)(fc)
    dense1 = tf.keras.layers.Dense(500, tf.nn.elu)(dense)
    point_pred = tf.keras.layers.Dense(2, activation=None, name="point_pred")(dense1)
    stroke_pred = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="stroke_pred")(dense1)
    model = tf.keras.models.Model(inputs=[main_input], outputs=[point_pred, stroke_pred])

    if os.listdir(p.log_root):
        model.load_weights(p.log_root)

    for i in range(50):
        fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
        draw(stk, ax=ax)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,)) / 127.5 - 1
        plt.close(fig)
        x = np.reshape(cv2.resize(data, (p.size, p.size)), [1, p.size, p.size, 3])
        pt, s = model.predict(x)
        sketch_ge = np.zeros((1, 3), dtype=np.float32)
        sketch_ge[:, :2] = pt * 100 + 8
        sketch_ge[:, 2:] = s * 100000 - 172532
        print(sketch_ge)
        stk = np.concatenate((stk, sketch_ge), 0)

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    draw(stk, ax=ax)
    plt.show()





