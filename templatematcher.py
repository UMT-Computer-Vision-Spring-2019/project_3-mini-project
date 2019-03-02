import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math as ma


def convolve(g, h, img):
    out = np.zeros((img.shape[0] - 2, img.shape[1] - 2))  # Output image has 2 less rows and columns from convolution

    # Simply loop through all pixels in the image and apply g and h to those pixels
    for u in range(1, out.shape[0] + 1):
        for v in range(1, out.shape[1] + 1):
            out[u - 1, v - 1] = g(u, v, img) * h(u, v, img, 0.0)

    return out


# Our h gauss function
def h_gauss(p, k, rad, sigma):
    val = p * p + k * k
    val = val / (2 * sigma * sigma)
    val = ma.exp(-val)

    return val * 1 / ((2 * rad + 1) ^ 2)


# A generic G function
def gn(u, v, img):
    return img[u, v]


# A generic convolution function
def convolve_gauss(g, h, img, rad, sigma):
    out = np.zeros((img.shape[0], img.shape[1]))

    ja = list(range(-rad, rad + 1))
    ka = list(range(-rad, rad + 1))

    # Simply loop through all pixels in the image and apply g and h to those pixels
    for u in range(0, img.shape[0]):
        for v in range(0, img.shape[1]):
            red_sum = 0.0

            for j in ja:
                for k in ka:
                    if 0 > u+j or u+j > img.shape[0] - 1:
                        continue
                    if 0 > v+k or v+k > img.shape[1] - 1:
                        continue
                    red_sum += g(u + j, v + k, img) * h(j, k, rad, sigma)

            out[u, v] = red_sum

    return out


# A function to create a series of reduced images
def create_image_pyramid(original_image):
    pyramid = []

    # find how many times we can divide the minimum dimension by two
    width = original_image.shape[0]
    height = original_image.shape[1]
    log_dim = ma.floor(ma.log2(min([width, height])))-1

    # of course we want to look for the original image
    pyramid.append(original_image)

    # append reduced images
    for n in range(log_dim):
        pyramid.append(reduced_gaussian_image(pyramid[n]))

    return pyramid


def reduced_gaussian_image(image):
    reduced_image = np.zeros((int((image.shape[0])/2), int(image.shape[1]/2)))
    # apply the filter
    g_img = convolve_gauss(gn, h_gauss, image, 1, 1)

    # for every pixel in reduced image, average the four pixels around the reduction
    for b in range(0, reduced_image.shape[0]):
        for c in range(0, reduced_image.shape[1]):
            reduced_image[b, c] = g_img[b*2, c*2]+g_img[b*2+1, c*2]+g_img[b*2, c*2+1]+g_img[b*2+1, c*2+1]/4.0
            
    return reduced_image


# with images always use try with resources. This ensures that the image closes when it's not longer in use.
with Image.open("waldo_template.jpg") as waldo_face:

    w = np.array(waldo_face)
    w = np.mean(w, -1)

    img_pyr = create_image_pyramid(w)

    for x in img_pyr:
        plt.imshow(x, cmap="Greys_r")
        plt.show()
