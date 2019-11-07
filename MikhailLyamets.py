from matplotlib import numpy as np
import matplotlib.pyplot as plt
import os

# from numba import njit

""" by Mikhail Lyamets
    @fenchelfen
    
    !!! If you'd like to speed up the code greatly,
        please import njit and revive the @njit decorators
    
    ``` If you'd like to see a nice visualisation,
        please uncomment show_images function at the end of main() 
"""


def show_images(images, names, scale=3):
    """
    Show an even number of images on a grid, use for debug
    :param images: np arrays
    :param names: images' names
    :param scale: how large your pictures should be
    """
    num_cols = len(images) // 2
    num_rows = len(images) // num_cols
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            # plt.imsave(os.path.join('/tmp/', images[index], cmap='', type='tiff')
            axes[i][j].imshow(images[index], cmap=plt.get_cmap('gray'))
            axes[i][j].set_title(names[index])
    plt.show()


def rgb2gray(image):
    """
    https://stackoverflow.com/a/12201744/9308909
    :param image: np array
    :return: grayscale image as np array
    """
    return image if len(image.shape) < 3 else \
        image[..., :3] @ [0.2989, 0.5870, 0.1140]


def sparsity_check(hat):
    """
    Visualize frequency image components so that the dominant ones are centered
    :param image: np array after fft2
    :return: shifted image filtered through log (to see the accents better)
    """
    return np.log(abs(np.fft.fftshift(hat) + 1))


# @njit
def rev(num, lg_n):
    """
    Help the original iterative fft compute certain black magic bitshifts
    """
    res = 0
    for i in range(lg_n):
        if num & (1 << i):
            res |= 1 << (lg_n - 1 - i)
    return res


# @njit
def fft2(image, invert=False):
    """
    Perform a two-dimensional fft in an iterative way
    :param image: np array
    :param invert: true if ifft
    :return: processed image as np array
    """
    shape = image.shape

    im = np.full(image.size, complex(0))
    for i, each in enumerate(image.reshape(-1)):
        im[i] = complex(each)

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
    return fft2(image, invert=True)


def main():
    """
    Iterate over tiff images in input/ and remove unwanted noise
    :return: 0 if success, 2 if reading error
    """
    root = './input'

    if not os.path.exists(root):
        print(f'no {root} folder is found, abort', end='\n\n')
        return 2

    _, _, names = list(os.walk(root))[0]
    images = []
    for each in names:
        path = os.path.join(root, each)
        image = plt.imread(path)
        images.append(rgb2gray(image))

    if not images:
        print(f'{root} is empty, abort', end='\n\n')
        return 2

    decompressed = []
    for each in images:
        hat = fft2(each)
        th = np.percentile(abs(hat), 70)
        truncated = np.where(abs(hat) > th, hat, hat * 0)
        truncated = fft2(truncated, True)
        decompressed.append(truncated)

    decompressed[:] = map(np.real, decompressed)

    OUT_DIR = 'MikhailLyametsOutputs'

    try:
        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)
    except OSError:
        print(f'Creation of the output directory {OUT_DIR} failed')

    for each, name in zip(decompressed, names):
        name, ext = os.path.splitext(name)
        plt.imsave(arr=each, fname=os.path.join(OUT_DIR, name + 'Compressed' + ext), format='tiff', cmap='gray')

    # show_images(images + decompressed, names + names, scale=5)


if __name__ == '__main__':
    main()
