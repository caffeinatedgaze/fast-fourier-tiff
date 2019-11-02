from matplotlib import numpy as np
import matplotlib.pyplot as plt
import os
from numba import njit


def show_images(images, names, scale=3):
    num_cols = len(images) // 2
    num_rows = len(images) - num_cols
    print(num_cols, num_rows)
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(images[i * num_cols + j], cmap=plt.get_cmap('gray'))
            axes[i][j].set_title(names[i * num_cols + j])
    plt.show()


def rgb2gray(image):
    """
    https://stackoverflow.com/a/12201744/9308909
    :param image:
    :return:
    """
    return image if len(image.shape) < 3 else \
        image[..., :3] @ [0.2989, 0.5870, 0.1140]


def sparsity_check(hat):
    """
    Convert your fft2 processed image to a fancy one
    :param image:
    :return:
    """
    return np.log(abs(np.fft.fftshift(hat) + 1))


@njit
def rev(num, lg_n):
    res = 0
    for i in range(lg_n):
        if num & (1 << i):
            res |= 1 << (lg_n - 1 - i)
    return res


def cfft2(image):
    return np.fft.fft2(image)


@njit
def fft2(image, invert=False):
    shape = image.shape
    im = np.copy(image.reshape(-1))
    length = len(im)
    lg_n = 0

    t = 1 << lg_n
    while t < length:
        lg_n += 1
        t = 1 << lg_n

    for i in range(length):
        if i < rev(i, lg_n):
            ind = rev(i, lg_n)
            im[i], im[ind] = im[ind], im[i]

    ln = 2
    while ln <= length:
        angle = 2 * np.pi * (-1 if invert else 1)
        wlen = np.complex(np.cos(angle), np.sin(angle))
        for i in range(0, length, ln):
            w = np.complex(1)
            for j in range(ln // 2):
                v = np.real(im[i + j + ln // 2] * w)
                u = np.complex(np.real(im[i + j]), v)
                im[i + j] = np.real(u + v)
                im[i + j + ln // 2] = np.real(u - v)
                w *= wlen
        ln <<= 1
    if invert:
        for i in range(length):
            im[i] /= length
    return im.reshape(shape)


def ifft2(image):
    """
    Perform a two-dimensional ifft
    :param image:
    :return:
    """
    return np.fft.ifft2(image)


def main():
    root = './input'
    _, _, names = list(os.walk(root))[0]
    images = []
    for each in names[:1]:
        path = os.path.join(root, each)
        image = plt.imread(path)
        images.append(rgb2gray(image))

    if not images:
        print('input/ is empty')
        return

    # show_images(images, names)

    # images = list(map(sparsity_check, images))
    # show_images(images, names)

    A_hat = list(map(fft2, images))

    decompressed = []
    print(len(A_hat))
    for each in A_hat:
        # thresholds = .1 * np.array([0.001, 0.005, 0.01]) * \
        #              np.max(np.abs(each))
        # print(thresholds)
        truncated = np.real(each)
        truncated = np.real(fft2(truncated.astype(complex), invert=True))
        # truncated = np.real(ifft2(truncated))
        decompressed.append(truncated)

    # show_images(decompressed, names)
    # show_images(images[:2] + decompressed[:2], names[:2] + names[:2], scale=10)

    show_images([images[0]] * 2 + [decompressed[0]] * 2, [names[0]] * 4)


if __name__ == '__main__':
    main()
