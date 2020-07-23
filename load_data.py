import vector_2_img as vg
import numpy as np
import matplotlib.pyplot as plt

FILE_PATH = 'sketches/'
test = np.load(FILE_PATH + 'cat.npy')
print(np.shape(test))
img1 = test[5,]
img1 = np.reshape(img1, (28, 28))
plt.imshow(img1)
plt.waitforbuttonpress(0)

