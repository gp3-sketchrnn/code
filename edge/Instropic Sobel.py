#!/usr/bin/env python
# coding: utf-8

# In[14]:


# -*- coding: utf-8 -*-
import cv2  
import numpy as np  
import matplotlib.pyplot as plt
 
#读取图像
img = cv2.imread('lena.jpg')
lenna_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
#Instropic Sobel算子
sq = 2**0.5
kernelx = np.array([[-1,-sq,-1],[0,0,0],[1,sq,1]], dtype=int)
kernely = np.array([[-1,0,1],[-sq,0,sq],[-1,0,1]], dtype=int)
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
#转uint8 
absX = cv2.convertScaleAbs(x)      
absY = cv2.convertScaleAbs(y)    
Instropic_Sobel = cv2.addWeighted(absX,0.5,absY,0.5,0)

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei']

#显示图形
titles = [u'原始图像', u'Instropic Sobel算子']  
images = [lenna_img, Roberts]  
for i in range(2):  
    plt.subplot(1,2,i+1), plt.imshow(images[i], 'gray')  
    plt.title(titles[i])  
    plt.xticks([]),plt.yticks([])  
plt.show()

cv2.imwrite('Instropic Sobel.jpg', Instropic_Sobel)


# In[16]:


Instropic_Sobel_inverse = 255 - Instropic_Sobel
plt.imshow(Instropic_Sobel_inverse,'gray')
plt.xticks([]),plt.yticks([])  
plt.show()
cv2.imwrite('Instropic_Sobel_invert.jpg', Instropic_Sobel_inverse)


# In[19]:


[m,n] = Instropic_Sobel_inverse.shape
im = np.zeros([m,n],dtype=np.uint8)
print(im.shape)
for i in range(m):
    for j in range(n):
        if(Instropic_Sobel_inverse[i][j]>200):
            im[i][j] = 255
        else:
            im[i][j] = 0
plt.imshow(im,'gray')
plt.xticks([]),plt.yticks([])  
plt.show()
cv2.imwrite('Instropic_Sobel_invert_thr.jpg', im)


# In[ ]:




