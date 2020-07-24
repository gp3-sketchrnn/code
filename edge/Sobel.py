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
 
#Sobel算子
x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0) #对x求一阶导
y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1) #对y求一阶导
absX = cv2.convertScaleAbs(x)      
absY = cv2.convertScaleAbs(y)    
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图形
titles = [u'原始图像', u'Sobel算子']  
images = [lenna_img, Sobel]  
for i in range(2):  
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
    plt.title(titles[i])  
    plt.xticks([]),plt.yticks([])  
plt.show()

cv2.imwrite('Sobel.jpg', Sobel)


# In[5]:


Sobel_inverse = 255 - Sobel
plt.imshow(Sobel_inverse,'gray')
plt.xticks([]),plt.yticks([])  
plt.show()
cv2.imwrite('Sobel_invert.jpg', Sobel_inverse)


# In[10]:


[m,n] = Sobel_inverse.shape
im = np.zeros([m,n],dtype=np.uint8)
print(im.shape)
for i in range(m):
    for j in range(n):
        if(Sobel_inverse[i][j]>180):
            im[i][j] = 255
        else:
            im[i][j] = 0
plt.imshow(im,'gray')
plt.xticks([]),plt.yticks([])  
plt.show()
cv2.imwrite('Sobel_invert_thr.jpg', im)


# In[ ]:




