import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numba
from numba import jit
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
def create_image_pyramid(original_image, cross_corr=False):
    t_pyr = []
    pyramid = []

    # find how many times we can divide the minimum dimension by two
    width = original_image.shape[0]
    height = original_image.shape[1]
    log_dim = ma.floor(ma.log2(min([width, height])))-4

    # of course we want to look for the original image
    t_pyr.append(original_image)

    # append reduced images
    for n in range(log_dim):
        t_pyr.append(reduced_gaussian_image(t_pyr[n]))
    # Lucas' code
    if cross_corr:
        for n in range(log_dim):
            continue
    else:
        # find z-score of template
        for n in t_pyr:
            i_mean = np.mean(n)
            i_std_dev = np.std(n)
            pyramid.append(z_normalize(n, i_mean, i_std_dev))

    return pyramid


# returns a normalized array
@jit(nopython=True)
def z_normalize(img, m, s_dev):
    normalized_array = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            normalized_array[i, j] = (img[i, j] - m) / s_dev
    return normalized_array


def reduced_gaussian_image(image):
    reduced_image = np.zeros((int((image.shape[0])/2), int(image.shape[1]/2)))
    # apply the filter
    g_img = convolve_gauss(gn, h_gauss, image, 1, 1)

    # for every pixel in reduced image, average the four pixels around the reduction
    for b in range(0, reduced_image.shape[0]):
        for c in range(0, reduced_image.shape[1]):
            reduced_image[b, c] = g_img[b*2, c*2]+g_img[b*2+1, c*2]+g_img[b*2, c*2+1]+g_img[b*2+1, c*2+1]/4.0

    return reduced_image


@jit(nopython=True)
def slide_image(target, template):
    dim1, dim2 = template.shape
    errors = []
    # iterate from zero to len(img)-len(template)
    for i in range(0, int(target.shape[0]-dim1)):
        for j in range(0, int(target.shape[1]-dim2)):
            img = target[i:i+dim1, j:j+dim2]
            errors.append(((i+(dim1/2), j+(dim2/2)), sse(img, template)))
    return errors


# returns the sum square error of two images of equal shape
@jit(nopython=True, parallel=True)
def sse(img1, img2):
    error = 0.0
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            error += np.square(img1[i, j]-img2[i, j])
    return error


# with images always use try with resources. This ensures that the image closes when it's not longer in use.
with Image.open("waldo_template.jpg") as waldo_face:

    w = np.array(waldo_face)
    w = np.mean(w, -1)

img_pyr = create_image_pyramid(w)

# for img in img_pyr:
#     plt.imshow(img, cmap="Greys_r")
#     plt.show()

with Image.open("waldo_1.jpg") as t:
    original = t
    targ = np.array(t)
    targ = np.mean(targ, -1)
# normalize target image
mean = np.mean(targ)
std_dev = np.std(targ)
targ = z_normalize(targ, mean, std_dev)

pred = []
for im in img_pyr:
    pred.append(min(slide_image(targ, im), key=lambda p: p[1]))
best_pred = min(pred, key=lambda p:p[1])
with Image.open("waldo_1.jpg") as t:
    plt.imshow(t)
# for e in pred:
#    plt.plot(e[0][1], e[0][0], 'og-', fillstyle="none", linewidth=4, markersize=15)

plt.plot(best_pred[0][1], best_pred[0][0], 'og', fillstyle="none", linewidth=5, markersize=15)
plt.savefig('filename.png', dpi=1600)
