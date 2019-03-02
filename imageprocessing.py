import numpy as np
import math as mt

def sum_squared_error(D1, D2):
    if(D1.shape == D2.shape):
        return np.sum((((D1-np.mean(D1))/np.std(D1)) - ((D2-np.mean(D2))/np.std(D2)))**2)
    else:
        return np.inf

def gauss_kernal(size, var):
       '''Returns a gauss kernal with the given size and variance'''
       
       kernel = np.zeros(shape=(size,size))
       for i in range(size):
              for j in range(size):
                     kernel[i][j] = mt.exp( -((i - (size-1)/2)**2 + (j - (size-1)/2)**2 )/(2*var*var))

              kernel = kernel / kernel.sum()
              
       return kernel

def convolve(g,h):
       '''Convolves image g with kernal h'''

       I_gray_copy = g.copy()
       x,y = h.shape
       xl = int(x/2)
       yl = int(y/2)
       for i in range(xl,len(g[:,1])-xl):
              for j in range(yl, len(g[i,:])-yl):

                     f = g[i-xl:i+(xl+1), j-yl:j+(yl+1)]

                     total = h*f
                     
                     I_gray_copy[i][j] = sum(sum(total))

       return I_gray_copy

def gauss_blur(image):
       '''Returns the image blurred with a gauss kernel of variance 1'''
       g_kernal = gauss_kernal(3,1)
       g_blur = convolve(image, g_kernal)
       return g_blur
