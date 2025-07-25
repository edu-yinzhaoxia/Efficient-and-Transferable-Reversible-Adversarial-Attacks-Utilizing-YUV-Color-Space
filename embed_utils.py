import math
import numpy as np
from helpers import *

def compressfun(signal):
    """Compress signal using arithmetic encoding."""
    L = len(signal)
    precision = 32
    signal = signal[:, 0].tolist()
    code, dic = Arithmetic_encode(signal, precision)
    return code

def crossPrediction(I):
    """Predict half the pixels (i+j even) and return predicted image, error map, and mask."""
    [m, n] = I.shape
    I1 = np.pad(I, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=(0, 0))
    p = np.zeros((I1.shape))
    Ic = I1.copy()
    for ii in range(1, m + 1):
        for jj in range(1, n + 1):
            if (ii + jj + 2) % 2 == 0:
                Ic[ii, jj] = math.floor((I1[ii + 1, jj] + I1[ii - 1, jj] + I1[ii, jj + 1] + I1[ii, jj - 1]) / 4)
                p[ii, jj] = 1
    Ic = Ic[1:-1, 1:-1]
    p = p[1:-1, 1:-1]
    er = I - Ic
    return Ic, er, p

def EmbeddingHistogramShifting(I_pred, data, T, er, p):
    """Embed data and perform histogram shifting."""
    [m, n] = I_pred.shape
    ed = np.zeros((m, n))
    temp = 0
    for ii in range(m):
        for jj in range(n):
            if p[ii, jj] == 1:
                if (er[ii, jj] >= (-T)) & (er[ii, jj] < T):
                    temp += 1
                    ed[ii, jj] = 2 * er[ii, jj] + data[temp - 1]
                elif er[ii, jj] >= T:
                    ed[ii, jj] = er[ii, jj] + T
                elif er[ii, jj] < (-T):
                    ed[ii, jj] = er[ii, jj] - T
    I_stego = I_pred + ed
    noBitsEmbedded = temp
    return I_stego, noBitsEmbedded

def dotPrediction(I):
    """Predict the other half of the pixels (i+j odd) for prediction-error embedding."""
    [m, n] = I.shape
    I1 = np.pad(I, pad_width=((1, 1), (1, 1)), mode='constant', constant_values=(0, 0))
    p = np.zeros((I1.shape))
    Id = I1.copy()
    for ii in range(1, m + 1):
        for jj in range(1, n + 1):
            if (ii + jj + 2) % 2 != 0:
                Id[ii, jj] = math.floor((I1[ii + 1, jj] + I1[ii - 1, jj] + I1[ii, jj + 1] + I1[ii, jj - 1]) / 4)
                p[ii, jj] = 1
    Id = Id[1:-1, 1:-1]
    p = p[1:-1, 1:-1]
    er = I - Id
    return Id, er, p

def calculate_threshold(I, data, length):
    """Adaptive selection of threshold T based on embedding data length."""
    T = 1
    [ICrossPred, ec, pc] = crossPrediction(I)
    while 1:
        [Ic, crossEC] = EmbeddingHistogramShifting(ICrossPred, data, T, ec, pc)
        [IDotPred, ed, pd] = dotPrediction(Ic)
        [Istego, dotEC] = EmbeddingHistogramShifting(IDotPred, data[crossEC:], T, ed, pd)
        totalEmbeddedData = crossEC + dotEC
        if totalEmbeddedData >= length:
            break
        T = T + 5
    return T

def PE_encode(I, T, data):
    """Prediction Error Expansion encoding (embedding)."""
    [ICrossPred, ec, pc] = crossPrediction(I)
    [Ic, crossEC] = EmbeddingHistogramShifting(ICrossPred, data, T, ec, pc)
    [IDotPred, ed, pd] = dotPrediction(Ic)
    [Istego, dotEC] = EmbeddingHistogramShifting(IDotPred, data[crossEC:], T, ed, pd)
    return Istego

def ExtractionHistogramShifting(Isteg, er, T, p):
    """Extract data and restore histogram after embedding."""
    [m, n] = Isteg.shape
    temp = 0
    dataRec = []
    e = np.zeros((m, n))
    for ii in range(m):
        for jj in range(n):
            if p[ii, jj] == 1:
                if (er[ii, jj] >= (-2*T)) & (er[ii, jj] < (2*T)):
                    data = er[ii, jj] % 2
                    dataRec.append(data)
                    e[ii, jj] = np.floor(er[ii, jj] / 2)
                elif er[ii, jj] < (-2*T):
                    e[ii, jj] = er[ii, jj] + T
                elif er[ii, jj] >= (2*T):
                    e[ii, jj] = er[ii, jj] - T
    Irec = Isteg + e
    return Irec, dataRec

def PE_decode(T, Istego):
    """Prediction Error Expansion decoding (extraction and recovery)."""
    [IDotPredExtract, edExtract, pd] = dotPrediction(Istego)
    [IDotRec, dataDotRec] = ExtractionHistogramShifting(IDotPredExtract, edExtract, T, pd)
    [ICrossPredExtract, ecExtract, pc] = crossPrediction(IDotRec)
    [Irec, dataCrossRec] = ExtractionHistogramShifting(ICrossPredExtract, ecExtract, T, pc)
    recoveredData = dataCrossRec + dataDotRec
    return Irec, recoveredData

def embed_main(ori_yuv, advy_yuv):
    """Embed the Y-channel difference between advy_yuv and ori_yuv into the UV channels."""
    ori_y = ori_yuv[:, :, 0]
    advy_y = advy_yuv[:, :, 0]
    err = np.zeros((299*299, 1))
    index = 0
    for i in range(299):
        for j in range(299):
            err[index] = ori_y[i, j] - advy_y[i, j]
            index += 1

    mess = compressfun(err)
    mess_s = mess
    L = len(mess)
    bpp = L / (advy_y.size) / 2

    img_cb = advy_yuv[:, :, 1]
    img_cr = advy_yuv[:, :, 2]
    img = np.vstack((img_cb, img_cr))
    img_s = img.copy()

    [m, n] = img.shape
    count = math.ceil(L / (m * n))
    T_flag = []

    for index in range(count):
        cover = img.copy()
        data = mess
        len_data = len(data)
        if len_data <= 0:
            break

        if len_data >= (m * n - 18):
            len_embed = bin(m * n - 18)[2:]
            len_embed_data = [0] * (18 - len(len_embed)) + [int(b) for b in len_embed]
            data = len_embed_data + data
            embed_data = data[:m * n]
            T = calculate_threshold(cover, embed_data, m * n)
            T_flag.append(T)
            mess = data[m * n:]
        else:
            len_embed = bin(len(data))[2:]
            len_embed_data = [0] * (18 - len(len_embed)) + [int(b) for b in len_embed]
            embed_data = len_embed_data + data
            embed_data += [0] * (m * n - len(embed_data))
            T = calculate_threshold(cover, embed_data, 18 + len(data))
            T_flag.append(T)
            mess = []
        Istego = PE_encode(cover, T, embed_data)
        img = Istego

    img_result = img.copy()
    return img_result
