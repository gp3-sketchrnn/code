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
    for i in range(1000):
        train_img = cv2.imread(p.train_path + str(i) + ".jpg")
        train_set.append(train_img)

    train_set = np.asarray(train_set, np.float32) / 127.5 - 1
    # train_set = rgb2bgr(train_set)
    label = np.asarray(np.load(p.label_path + "label.npy"), dtype=np.float32)
    return train_set, label


if __name__ == '__main__':
    t_x, t_y = get_data()
    train_ds = (_input_fn(t_x[:900], t_y[:900], 900))
    validation_ds = _input_fn(t_x[900:], t_y[900:], 100)
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

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3, beta_1=0.5, decay=1e-5),
                  loss=tf.keras.losses.Huber(delta=.5),
                  metrics=["accuracy"])

    history = model.fit(train_ds, epochs=50, steps_per_epoch=10, validation_steps=10, validation_data=validation_ds)

    validation_steps = 20

    loss0, accuracy0 = model.evaluate(train_ds, steps=validation_steps)
    model.save_weights(p.log_root)

    print("loss: {:.2f}".format(loss0))
    print("accuracy: {:.2f}".format(accuracy0))
