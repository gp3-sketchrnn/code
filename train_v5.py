import tensorflow as tf
import hparam as p
import numpy as np
import cv2
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


def _input_fn(x, y, data_size):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=data_size)
    ds = ds.repeat()
    ds = ds.batch(p.batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_data():
    train_set = list()
    for idx in range(10000):
        train_img = cv2.imread(p.train_path + str(idx) + ".jpg")
        train_set.append(train_img)

    train_set = 1 - np.asarray(train_set) / 255.0
    print(np.shape(train_set))
    # train_set = rgb2bgr(train_set)
    label = np.asarray(np.load(p.label_path + "label.npy"), dtype=np.float32)
    return train_set, label


def get_mixture_coef(output):
    """Returns the tf slices containing mdn dist params."""
    # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
    z = output
    z_pen_logits = z[:, 0:3]  # pen states
    z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

    # process output z's into MDN parameters

    # softmax all the pi's and pen states:
    z_pi = tf.nn.softmax(z_pi)
    z_pen = tf.nn.softmax(z_pen_logits)

    # exponentiate the sigmas and also make corr between -1 and 1.
    z_sigma1 = tf.exp(z_sigma1)
    z_sigma2 = tf.exp(z_sigma2)
    z_corr = tf.tanh(z_corr)

    r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
    return r


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
    norm1 = tf.compat.v1.subtract(x1, mu1)
    norm2 = tf.compat.v1.subtract(x2, mu2)
    s1s2 = tf.compat.v1.multiply(s1, s2)
    # eq 25
    z = (tf.compat.v1.square(tf.compat.v1.div(norm1, s1)) + tf.compat.v1.square(tf.compat.v1.div(norm2, s2)) -
         2 * tf.compat.v1.div(tf.compat.v1.multiply(rho, tf.compat.v1.multiply(norm1, norm2)), s1s2))
    neg_rho = 1 - tf.compat.v1.square(rho)
    result = tf.exp(tf.compat.v1.div(-z, 2 * neg_rho))
    denom = 2 * np.pi * tf.compat.v1.multiply(s1s2, tf.compat.v1.sqrt(neg_rho))
    result = tf.compat.v1.div(result, denom)
    return result


def get_loss_function(y, y_pred):
    target = tf.reshape(y, [-1, 5])
    [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(target, 5, 1)
    pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)
    out = get_mixture_coef(y_pred)
    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits] = out
    return compute_loss(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen_logits, x1_data, x2_data, pen_data)


def compute_loss(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                 z_pen_logits, x1_data, x2_data, pen_data):
    """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
    # This represents the L_R only (i.e. does not include the KL loss term).

    result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
    epsilon = 1e-6
    # result1 is the loss wrt pen offset (L_s in equation 9 of
    # https://arxiv.org/pdf/1704.03477.pdf)
    result1 = tf.multiply(result0, z_pi)
    result1 = tf.compat.v1.reduce_sum(result1, 1, keep_dims=True)
    result1 = -tf.compat.v1.log(result1 + epsilon)  # avoid log(0)

    fs = 1.0 - pen_data[:, 2]  # use training data for this
    fs = tf.reshape(fs, [-1, 1])
    # Zero out loss terms beyond N_s, the last actual stroke
    result1 = tf.multiply(result1, fs)

    # result2: loss wrt pen state, (L_p in equation 9)
    result2 = tf.nn.softmax_cross_entropy_with_logits(
        labels=pen_data, logits=z_pen_logits)
    result2 = tf.reshape(result2, [-1, 1])
    if not p.is_training:  # eval mode, mask eos columns
        result2 = tf.multiply(result2, fs)

    result = result1 + result2
    return result


def build_model():
    main_input = tf.keras.Input((p.size, p.size, 3), name="ipt")

    cv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[2, 2], strides=1, activation=tf.nn.relu, padding='same')(main_input)
    con = tf.keras.layers.Conv2D(filters=128, kernel_size=[2, 2], strides=2, activation=tf.nn.relu, padding='same')(cv1)

    cv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=[2, 2], strides=1, activation=tf.nn.relu, padding='same')(con)
    cv4 = tf.keras.layers.Conv2D(filters=256, kernel_size=[2, 2], strides=2, activation=tf.nn.relu, padding='same')(cv3)

    cv5 = tf.keras.layers.Conv2D(filters=512, kernel_size=[2, 2], strides=2, activation=tf.nn.relu, padding='same')(cv4)
    cv6 = tf.keras.layers.Conv2D(filters=512, kernel_size=[2, 2], strides=2, activation=tf.nn.relu, padding='same')(cv5)

    # cv7 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, padding='same')(cv6)
    # cv8 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=2, activation=tf.nn.relu, padding='same')(cv7)

    fc = tf.keras.layers.Flatten()(cv6)
    d1 = tf.keras.layers.Dense(2048, activation=tf.nn.elu)(fc)
    dp2 = tf.keras.layers.Dropout(0.2)(d1)
    d2 = tf.keras.layers.Dense(2048, activation=None)(dp2)
    opt = tf.keras.layers.Dense(123, activation=None)(d2)
    model = tf.keras.models.Model(inputs=[main_input], outputs=[opt])
    model.summary()

    if os.listdir(p.log_root):
        model.load_weights(p.log_root)

    return model


if __name__ == '__main__':
    r_x, r_y = get_data()
    train_ds = (_input_fn(r_x[:9000], r_y[:9000], 9000))
    validation_ds = _input_fn(r_x[9000:], r_y[9000:], 1000)

    model = build_model()

    for i in range(1, 10):
        lr = p.lr

        model.compile(optimizer=tf.keras.optimizers.Adam(lr, decay=1e-8),
                      loss=get_loss_function, metrics=["accuracy"])
        history = model.fit(train_ds, shuffle=True, epochs=100, steps_per_epoch=5, validation_data=validation_ds,
                            validation_steps=10, batch_size=p.batch_size)

        model.save_weights(p.log_root)
