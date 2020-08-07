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

    train_set = 1 - np.asarray(train_set, np.float32) / 255.0
    # train_set = rgb2bgr(train_set)
    label = np.asarray(np.load(p.label_path + "label.npy"), dtype=np.float32)
    return train_set, label


def build_model():
    main_input = tf.keras.Input((p.size, p.size, 3), name="ipt")

    cv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[2, 2], strides=1, activation=tf.nn.relu, padding='same')(
        main_input)
    cv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[2, 2], strides=1, activation=tf.nn.relu, padding='same')(cv1)
    pool1 = tf.keras.layers.MaxPool2D([2, 2], 2)(cv2)

    cv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, padding='same')(
        pool1)
    cv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, padding='same')(cv3)
    pool2 = tf.keras.layers.MaxPool2D([2, 2], 2)(cv4)

    cv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, padding='same')(
        pool2)
    cv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, padding='same')(cv5)
    pool3 = tf.keras.layers.MaxPool2D([2, 2], 2)(cv6)

    cv7 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, padding='same')(
        pool3)
    cv8 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], strides=1, activation=tf.nn.relu, padding='same')(cv7)
    pool4 = tf.keras.layers.MaxPool2D([2, 2], 2)(cv8)

    fc = tf.keras.layers.Flatten()(pool4)
    d1 = tf.keras.layers.Dense(1000, tf.nn.elu, use_bias=False)(fc)
    d2 = tf.keras.layers.Dense(1000, tf.nn.elu, use_bias=False)(d1)
    point_pred = tf.keras.layers.Dense(2, activation=None, name="point_pred", use_bias=False)(d2)
    model = tf.keras.models.Model(inputs=[main_input], outputs=[point_pred])
    model.summary()

    if os.listdir(p.log_root):
        model.load_weights(p.log_root)

    return model


if __name__ == '__main__':
    r_x, r_y = get_data()
    train_ds = (_input_fn(r_x[:9000], r_y[:9000], 9000))
    validation_ds = _input_fn(r_x[9000:], r_y[9000:], 1000)

    model = build_model()

    for i in range(1, 100):
        lr = (p.lr - p.mlr) * (p.dr ** i) + p.mlr

        model.compile(optimizer=tf.keras.optimizers.Adam(lr, decay=1e-8),
                      loss=tf.keras.losses.mean_squared_error, metrics=["accuracy"])

        model.fit(train_ds, shuffle=True, epochs=100, steps_per_epoch=10, validation_data=validation_ds,
                  validation_steps=10, batch_size=p.batch_size)

        model.save_weights(p.log_root)
