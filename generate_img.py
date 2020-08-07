import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.path import Path
import hparam as p

from magenta.models.sketch_rnn.sketch_rnn_train import *
from magenta.models.sketch_rnn.model import *
from magenta.models.sketch_rnn.utils import *

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
        codes = np.roll(self.to_code(data[::, -1].astype(int)), shift=1)
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


def decode(z_input=None, temperature=.1, factor=.2):
    z = None
    if z_input is not None:
        z = [z_input]
    sample_strokes, m = sample(
        sess,
        sample_model,
        seq_len=eval_model.hps.max_seq_len,
        temperature=temperature, z=z)
    return to_normal_strokes(sample_strokes)


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

raw_sketches = list()
re_sketches = list()
for index in range(1000):
    sketch = test_set.random_sample()
    # print(np.shape(sketch))
    # s_size = np.shape(sketch)[0]
    # padding_array = np.zeros([250, 2])
    # padding_array[0:s_size, 0:2] = sketch[:, :2]
    # re_sketches.append(np.reshape(padding_array, [500]))

    # z = encode(sketch)
    # sketch_reconstructed = decode(z, temperature=0.1)
    # raw_sketches.append(sketch_reconstructed)
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(xticks=[], yticks=[], frame_on=False))
    draw(sketch, ax=ax)
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    img2 = cv2.resize(data, (p.size, p.size))
    cv2.imwrite("data/cats/" + str(index) + ".jpg", img2)

    print(index)

print(np.shape(raw_sketches))

