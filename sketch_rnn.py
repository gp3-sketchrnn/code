import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.animation import FuncAnimation
from matplotlib.path import Path
from matplotlib import rc

import hparam as p
import train_v5 as tv5
import train_v4 as tv4

from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *
from magenta.models.sketch_rnn.rnn import *

import cv2
import os

rc('animation', html='html5')
np.set_printoptions(precision=8,
                    edgeitems=6,
                    linewidth=200,
                    suppress=True)
tf.logging.info("TensorFlow Version: {}".format(tf.__version__))


def load_env_compatible(data_dir, model_dir):
    """Loads environment for inference mode, used in jupyter notebook."""
    # modified https://github.com/tensorflow/magenta/blob/master/magenta/models/sketch_rnn/sketch_rnn_train.py
    # to work with depreciated tf.HParams functionality
    model_params = sketch_rnn_model.get_default_hparams()
    with tf.gfile.Open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
        data = json.load(f)
    fix_list = ['conditional', 'is_training', 'use_input_dropout', 'use_output_dropout', 'use_recurrent_dropout']
    for fix in fix_list:
        data[fix] = (data[fix] == 1)
    model_params.parse_json(json.dumps(data))
    return load_dataset(data_dir, model_params, inference_mode=True)


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


def encode(input_strokes):
    strokes = to_big_strokes(input_strokes, max_len=eval_hps_model.max_seq_len).tolist()
    strokes.insert(0, [0, 0, 1, 0, 0])
    seq_len = [len(input_strokes)]
    sz = sess.run(eval_model.batch_z, feed_dict={
        eval_model.input_data: [strokes], eval_model.sequence_lengths: seq_len})[0]
    return sz


def decode(cnn_model, z_input=None, temperature=.1, factor=.2):
    z = None
    if z_input is not None:
        z = [z_input]
    sample_strokes, re_strokes, m = sample(
        sess,
        sample_model,
        cnn_model,
        seq_len=eval_model.hps.max_seq_len,
        temperature=temperature, z=z)
    return to_normal_strokes(sample_strokes), to_normal_strokes(re_strokes)


DATA_DIR = '/tmp/sketch_rnn/dataset/'
MODELS_ROOT_DIR = '/tmp/sketch_rnn/models'

download_pretrained_models(models_root_dir=MODELS_ROOT_DIR, pretrained_models_url=PRETRAINED_MODELS_URL)

MODEL_DIR = MODELS_ROOT_DIR + '/catbus/lstm'

tf.disable_eager_execution()
[train_set, valid_set, test_set, hps_model, eval_hps_model, sample_hps_model] = load_env_compatible(DATA_DIR, MODEL_DIR)

reset_graph()
model = Model(hps_model)
eval_model = Model(eval_hps_model, reuse=True)
sample_model = Model(sample_hps_model, reuse=True)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

load_checkpoint(sess=sess, checkpoint_path=MODEL_DIR)

# ========   train   ==============
# main(None)

# ========   evaluate   ===========

# if not os.path.exists("data/results/sketch.npy"):
#     sketch = test_set.random_sample()
#     np.save("data/results/sketch", sketch)
# else:
#     sketch = np.load("data/results/sketch.npy")

# ========   cnn   ============
cnn_model = tv5.build_model()
# cnn_model = tv4.build_model()
# ========   cnn   ============

# for index in range(1000):
sketch = test_set.random_sample()
z = encode(sketch)
s_sketch, re_sketch = decode(cnn_model, z, temperature=0.05)

# fig, ax = plt.subplots(figsize=(1.28, 1.28), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
# draw(s_sketch, ax=ax)
# fig.canvas.draw()
# data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# plt.close(fig)
# img = cv2.resize(data, (128, 128))
# # cv2.imwrite("data/1/" + str(index) + ".jpg", img)
#
# fig, ax = plt.subplots(figsize=(1.28, 1.28), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
# draw(re_sketch, ax=ax)
# fig.canvas.draw()
# data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# plt.close(fig)
# img = cv2.resize(data, (128, 128))
# cv2.imwrite("data/2/" + str(index) + ".jpg", img)
# # print(index)

# fig, ax_arr = plt.subplots(nrows=5, ncols=10, figsize=(8, 4), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
# fig.tight_layout()
#
# for row_num, ax_row in enumerate(ax_arr):
#     for col_num, ax in enumerate(ax_row):
#         if not col_num:
#             draw(sketch, ax=ax)
#             xlabel = 'original'
#         else:
#             t = col_num / 10.
#             draw(decode(z, temperature=t), ax=ax)
#             xlabel = r'$\tau={}$'.format(t)
#         if row_num + 1 == len(ax_arr):
#             ax.set_xlabel(xlabel)
#
# plt.show()
#
# data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
# cv2.imwrite("data/results/random.jpg", data)

fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(6, 3), subplot_kw=dict(xticks=[], yticks=[]))
fig.tight_layout()

x_pad, y_pad = 10, 10

x_pad //= 2
y_pad //= 2

(x_min_1, x_max_1, y_min_1, y_max_1) = get_bounds(data=s_sketch, factor=.2)

(x_min_2, x_max_2, y_min_2, y_max_2) = get_bounds(data=re_sketch, factor=.2)

x_min = np.minimum(x_min_1, x_min_2)
y_min = np.minimum(y_min_1, y_min_2)

x_max = np.maximum(x_max_1, x_max_2)
y_max = np.maximum(y_max_1, y_max_2)

ax1.set_xlim(x_min - x_pad, x_max + x_pad)
ax1.set_ylim(y_max + y_pad, y_min - y_pad)

ax1.set_xlabel('Original')

ax2.set_xlim(x_min - x_pad, x_max + x_pad)
ax2.set_ylim(y_max + y_pad, y_min - y_pad)

ax2.set_xlabel('Reconstruction')


def animate(i):
    original = SketchPath(s_sketch[:i + 1])
    reconstructed = SketchPath(re_sketch[:i + 1])

    patch1 = ax1.add_patch(patches.PathPatch(original,
                                             facecolor='none'))

    patch2 = ax2.add_patch(patches.PathPatch(reconstructed,
                                             facecolor='none'))

    return patch1, patch2


frames = np.maximum(s_sketch.shape[0], re_sketch.shape[0])
FuncAnimation(fig, animate, frames=frames - 1, interval=15, repeat_delay=1000 * 3, blit=True)
plt.show()
