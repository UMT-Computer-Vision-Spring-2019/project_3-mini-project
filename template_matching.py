import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import cv2
import scipy

I_1 = plt.imread('waldo_1.jpg')

waldo = plt.imread('waldo_template.jpg')

error = float('inf')
min_u = 0
min_v = 0
min_size = ((0,0))
while (waldo.shape[0] > 15):
  waldo_width = waldo.shape[0]
  waldo_height = waldo.shape[1]
  for u in range(0, I_1.shape[0] - waldo_width):
    for v in range(0, I_1.shape[1] - waldo_height):
      patch = I_1[u:u + waldo_width, v: v + waldo_height]
      test_error = np.sum((patch-waldo)**2)
      if test_error < error:
        min_u = u
        min_v = v
        min_size = ((waldo_width, waldo_height))
        error = test_error      
  print(waldo.shape)
  waldo = gaussian_filter(waldo, sigma=1)
  waldo = scipy.misc.imresize(waldo, 0.5)
print(min_u, min_v, min_size)
plt.imshow(I_1)
plt.show()
