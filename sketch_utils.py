import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.path import Path
from magenta.models.sketch_rnn.utils import *
import cv2
import tensorflow as tf
import hparam as p
from tensorflow.keras import backend as k


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
    fig, ax = plt.subplots(figsize=(1.28, 1.28), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    draw(sketch_reconstructed, ax=ax)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    img = cv2.resize(data, (p.size, p.size))
    return np.reshape(img, (1, p.size, p.size, 3))


def get_suggested_point(cnn_model, img):
    images = 1 - img / 255.0
    prob = cnn_model.predict(images)
    # get_layer_output = k.function([cnn_model.layers[0].input], [cnn_model.layers[5].output])
    # layer_output = get_layer_output(images)[0]
    # layer_output = np.reshape(layer_output, [16, 16, 512])
    # graph = np.zeros([16 * 16, 16 * 32])
    # acc = 0
    # for i in range(16):
    #     for j in range(32):
    #         graph[i * 16: i * 16 + 16, j * 16: j * 16 + 16] = layer_output[:, :, acc]
    #         acc += 1
    # plt.imshow(graph)
    # plt.waitforbuttonpress(1)
    z = prob
    z_pen_logits = z[:, 0:3]  # pen states
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = np.split(z[:, 3:], 6, 1)

    z_pi = np.exp(z_pi) / np.sum(np.exp(z_pi))
    z_pen = np.exp(z_pen_logits) / np.sum(np.exp(z_pen_logits))

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = np.exp(z_sigma1)
    z_sigma2 = np.exp(z_sigma2)
    z_corr = np.tanh(z_corr)
    return z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen


def get_point(cnn_model, img):
    images = 1 - img / 255.0

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="model/callback")
    prob = cnn_model.predict(images, callbacks=[tb_callback])
    return prob[0][0], prob[0][1]
