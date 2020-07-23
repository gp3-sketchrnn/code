import numpy as np
import tensorflow.compat.v1 as tf

import vgg16
import utils
import time

img1 = utils.load_image("./test_img/tiger.jpeg")
img2 = utils.load_image("./test_img/puzzle.jpeg")

batch1 = img1.reshape((1, 32, 32, 3))
batch2 = img2.reshape((1, 32, 32, 3))

batch = np.concatenate((batch1, batch2), 0)

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", [2, 32, 32, 3])
    feed_dict = {images: batch}

    vgg = vgg16.Vgg16("/tmp/vgg16/vgg16.npy")
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    t1 = time.time() * 1000
    prob = sess.run(vgg.prob, feed_dict=feed_dict)
    t2 = time.time() * 1000
    print(t2 - t1)
