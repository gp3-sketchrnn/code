import numpy as np
import tensorflow.compat.v1 as tf

import vgg16
import tmp_model
import time

img1 = tmp_model.load_image("./test_img/tiger.jpeg")
img2 = tmp_model.load_image("./test_img/puzzle.jpeg")

batch1 = img1.reshape((1, 128, 128, 3))
batch2 = img2.reshape((1, 128, 128, 3))

batch = np.concatenate((batch1, batch2), 0)

with tf.Session(
        config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
    images = tf.placeholder("float", [2, 128, 128, 3], name="pc_img")
    feed_dict = {images: batch}
    noise = np.random.random([2, 128, 128, 3])

    vgg = vgg16.Vgg16("/tmp/vgg16/vgg16.npy")
    with tf.name_scope("content_vgg"):
        vgg.build(images)

    t1 = time.time() * 1000
    prob = sess.run(vgg.prob, feed_dict={images: noise})
    prob2 = sess.run(vgg.prob, feed_dict={images: batch})
    print(prob)
    print(prob2)
    t2 = time.time() * 1000
    print(t2 - t1)
