import inspect
import os

import numpy as np
import tensorflow.compat.v1 as tf
import time
import hparam as p

VGG_MEAN = [103.939, 116.779, 123.68]


def rgb2bgr(rgb):
    rgb_scaled = rgb
    # Convert RGB to BGR
    red, green, blue = tf.split(rgb_scaled, 3, 3)
    bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], 3)

    return bgr


def reset_graph():
    """Closes the current default session and resets the graph."""
    sess = tf.get_default_session()
    if sess:
        sess.close()
    tf.reset_default_graph()


def init(batch):
    vgg = Vgg16("/tmp/vgg16/vgg16.npy")
    with tf.name_scope("content_vgg"):
        vgg.build(batch)

    return vgg


def train(batch, train_set, label, log_root, resume_training):
    # reset_graph()
    model = init(batch)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    if resume_training:
        load_checkpoint(sess, log_root)

    start = time.time()

    for stp in range(p.num_steps):

        step = sess.run(model.global_step)

        curr_learning_rate = ((p.lr - p.mlr) * p.dr ** step + p.mlr)

        s_idx = 0 if batch * (stp + 1) >= 1000 else batch * stp

        feed = {
            model.input_data: train_set[s_idx: s_idx + batch],
            model.label: label[s_idx: s_idx + batch],
            model.lr: curr_learning_rate,
        }

        train_cost, _ = sess.run([model.cost, model.train_op], feed)

        if step % 20 == 0 and step > 0:
            end = time.time()
            time_taken = end - start

            cost_summ = tf.summary.Summary()
            cost_summ.value.add(tag='Train_Cost', simple_value=float(train_cost))
            lr_summ = tf.summary.Summary()
            lr_summ.value.add(tag='Learning_Rate', simple_value=float(curr_learning_rate))
            time_summ = tf.summary.Summary()
            time_summ.value.add(tag='Time_Taken_Train', simple_value=float(time_taken))

            output_format = 'step: %d, lr: %.6f, cost: %.4f, train_time_taken: %.4f'
            output_values = (step, curr_learning_rate, train_cost, time_taken)
            output_log = output_format % output_values

            tf.logging.info(output_log)

            start = time.time()

        if step % p.save_every == 0 and step > 0:
            save_model(sess, log_root, step)


def load_checkpoint(sess, checkpoint_path):
    saver = tf.train.Saver(tf.global_variables(scope="content_vgg"))
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


def save_model(sess, model_save_path, global_step):
    saver = tf.train.Saver(tf.global_variables(scope="content_vgg"))
    checkpoint_path = os.path.join(model_save_path, 'vector')
    tf.logging.info('saving model %s.', checkpoint_path)
    tf.logging.info('global_step %i.', global_step)
    saver.save(sess, checkpoint_path, global_step=global_step)


class Vgg16:
    def __init__(self, vgg16_npy_path=None):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.Variable(p.lr, trainable=False)
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "vgg16.npy")
            vgg16_npy_path = path
            print(path)
        # 加载网络权重参数
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        fc_prob = [np.random.normal(0, 0.001, [4096, 500]), np.random.normal(0, 0.001, [500, ])]
        self.data_dict['fc_prob'] = fc_prob
        print("npy file loaded")

    def build(self, batch):
        print("build model started")
        tf.disable_eager_execution()
        self.input_data = tf.placeholder(dtype=tf.float32, shape=[batch, p.size, p.size, 3])

        self.conv1_1 = self.conv_layer(self.input_data, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, "fc6")

        self.prob = self.fc_layer(self.fc6, "fc_prob")

        self.data_dict = None

        r = tf.reshape(self.prob, [batch, 250, 2])
        self.label = tf.placeholder(dtype=tf.float32, shape=[batch, 250, 2])
        self.cost = tf.reduce_mean((r - self.label) ** 2) / 2.0
        if p.is_training:
            self.lr = tf.Variable(p.lr, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)
            gvs = optimizer.compute_gradients(self.cost)
            g = 1.0
            capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs, global_step=self.global_step, name='train_step')

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.Variable(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.Variable(self.data_dict[name][1], name="biases", dtype=tf.float32)

    def get_fc_weight(self, name):
        return tf.Variable(self.data_dict[name][0], name="weights", dtype=tf.float32)
