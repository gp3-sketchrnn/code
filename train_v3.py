import tensorflow as tf
import hparam as p
import numpy as np
import cv2
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_data():
    train_set = list()
    for i in range(5000):
        train_img = cv2.imread(p.train_path + str(i) + ".jpg")
        train_set.append(train_img)

    train_set = np.asarray(train_set, np.float32) / 127.5 - 1
    # train_set = rgb2bgr(train_set)
    label = np.asarray(np.load(p.label_path + "label.npy"), dtype=np.float32)
    return train_set, label


if __name__ == '__main__':
    r_x, r_y = get_data()
    t_x, t_y = r_x[:900], r_y[:900]
    v_x, v_y = r_x[900:], r_y[900:]
    IMG_SHAPE = (p.size, p.size, 3)
    main_input = tf.keras.Input((p.size, p.size, 3), name="ipt")
    VGG16_MODEL = tf.keras.applications.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
    VGG16_MODEL.trainable = True
    vgg_result = VGG16_MODEL(main_input)
    fc = tf.keras.layers.Flatten()(vgg_result)
    dense = tf.keras.layers.Dense(1000, activation=tf.nn.elu)(fc)
    dense1 = tf.keras.layers.Dense(500, tf.nn.elu)(dense)
    point_pred = tf.keras.layers.Dense(2, activation=None, name="point_pred")(dense1)
    stroke_pred = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, name="stroke_pred")(dense1)
    model = tf.keras.models.Model(inputs=[main_input], outputs=[point_pred, stroke_pred])
    model.summary()

    if os.listdir(p.log_root):
        model.load_weights(p.log_root)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-2, decay=1e-8),
                  loss={'point_pred': tf.keras.losses.mean_squared_error,
                        'stroke_pred': tf.keras.losses.binary_crossentropy},
                  loss_weights={'point_pred': 0.5, 'stroke_pred': 0.5},
                  metrics=["accuracy"])

    history = model.fit(x={'ipt': t_x}, y={'point_pred': t_y[:, 0:2], 'stroke_pred': t_y[:, 2]}, epochs=50,
                        steps_per_epoch=10, validation_steps=10, batch_size=p.batch_size)

    validation_steps = 20

    # loss0, accuracy0 = model.evaluate(x={'ipt': v_x}, y={'point_pred': v_y[:, 0:2], 'stroke_pred': v_y[:, 2]},
    #                                   steps=validation_steps)
    model.save_weights(p.log_root)

