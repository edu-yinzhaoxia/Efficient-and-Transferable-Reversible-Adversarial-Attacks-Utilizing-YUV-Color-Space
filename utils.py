import numpy as np

def rgb2yuv(image):
    image = image[0].cpu()
    npimg = 255 * (image.numpy())
    img = np.transpose(npimg, (1, 2, 0))

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    yuv = img.copy()
    yuv[:, :, 0] = np.round(0.299 * R + 0.587 * G + 0.114 * B)
    yuv[:, :, 1] = np.round(- 0.1687 * R - 0.3313 * G + 0.5 * B + 128)
    yuv[:, :, 2] = np.round(0.5 * R - 0.4187 * G - 0.0813 * B + 128)

    return yuv


def yuv2rgb(yuv):
    Y = yuv[:, :, 0]
    U = yuv[:, :, 1]
    V = yuv[:, :, 2]

    R = np.maximum(0, np.minimum(255, np.round(Y + 1.402 * (V - 128))))
    G = np.maximum(0, np.minimum(255, np.round(Y - 0.34414 * (U - 128) - 0.71414 * (V - 128))))
    B = np.maximum(0, np.minimum(255, np.round(Y + 1.772 * (U - 128))))

    rgb = yuv.copy()
    rgb[:, :, 0] = R
    rgb[:, :, 1] = G
    rgb[:, :, 2] = B

    return rgb
