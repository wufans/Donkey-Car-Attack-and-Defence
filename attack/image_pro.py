"""
处理图片维度，生成噪声图片
"""
import numpy as np
import matplotlib.image as mpimg
import skimage
from skimage import color,io
import matplotlib.pyplot as plt
from PIL import Image

#[461,501,3] to [120,160,3]
img = Image.open("3.jpg")
width = 120
height = 160
dim = 3
out = img.resize((width,height),Image.ANTIALIAS)
out.save("3.jpg")
#plt.show()

#构建噪声图片
# img = mpimg.imread("test_img/stop.jpg")
# img2 =skimage.util.random_noise(img, mode='s&p', seed=None, clip=True)
# print(img.shape)
# plt.imshow(img)
# #io.imsave('stop_noise_s&p.jpg',img2)
