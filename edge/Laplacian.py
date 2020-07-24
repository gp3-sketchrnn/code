#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-
import cv2  
import numpy as np  
import matplotlib.pyplot as plt
 
#读取图像
img = cv2.imread('lena.jpg')
lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#拉普拉斯算法
dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize = 3)
Laplacian = cv2.convertScaleAbs(dst) 

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图形
titles = [u'原始图像', u'Laplacian算子']  
images = [lenna_img, Laplacian]  
for i in range(2):  
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
    plt.title(titles[i])  
    plt.xticks([]),plt.yticks([])  
plt.show()

cv2.imwrite('Laplacian.jpg', Laplacian)


# In[5]:


Laplacian_inverse = 255 - Laplacian
plt.imshow(Laplacian_inverse,'gray')
plt.xticks([]),plt.yticks([])  
plt.show()
cv2.imwrite('Laplacian_invert.jpg', Laplacian_inverse)


# In[7]:


[m,n] = Laplacian_inverse.shape
im = np.zeros([m,n],dtype=np.uint8)
print(im.shape)
for i in range(m):
    for j in range(n):
        if(Laplacian_inverse[i][j]>170):
            im[i][j] = 255
        else:
            im[i][j] = 0
plt.imshow(im,'gray')
plt.xticks([]),plt.yticks([])  
plt.show()
cv2.imwrite('Laplacian_invert_thr.jpg', im)


# In[ ]:




