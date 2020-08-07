import hparam as p
import numpy as np
from vgg16 import *
import cv2


if __name__ == '__main__':
    train_set = list()
    label = list()
    for i in range(1000):
        train_img = cv2.imread(p.train_path + str(i) + ".jpg")
        train_set.append(train_img)

    train_set = np.asarray(train_set, np.float32)
    # train_set = rgb2bgr(train_set)
    label = np.load(p.label_path + "label.npy")

    train(20, train_set, label, log_root=p.log_root, resume_training=False)



