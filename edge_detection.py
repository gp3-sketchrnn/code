import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature
import hparam as p
import numpy as np
import cv2
import os


cat_list = os.listdir("E:\\tmp\\cats\\CAT_00\\")
for c_name in cat_list:
    if c_name.endswith(".jpg"):
        img = cv2.imread("E:\\tmp\\cats\\CAT_00\\" + c_name, cv2.IMREAD_GRAYSCALE)
        size = np.shape(img)

        # if size[0] >= p.size:
        #     s1 = (size[0] - p.size) / 2
        #     img = img[s1:s1 + p.size, :]
        # if size[1] >= p.size:
        #     s1 = (size[1] - p.size) / 2
        #     img = img[:, s1:s1 + p.size]
        # if ()
        edges1 = feature.canny(img)
        edges2 = feature.canny(img, sigma=3)

        # display results
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('noisy image', fontsize=20)

        ax2.imshow(edges1)
        ax2.axis('off')
        ax2.set_title('Canny filter, $\sigma=1$', fontsize=20)

        ax3.imshow(edges2)
        ax3.axis('off')
        ax3.set_title('Canny filter, $\sigma=3$', fontsize=20)

        fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                            bottom=0.02, left=0.02, right=0.98)

        plt.show()

