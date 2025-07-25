import skimage.metrics as skmetrics
import numpy as np
import cv2 as cv

def calMetrics(ori_img,rae_img):
    # (bs,ch,h,w)
    ori_img,rae_img = ori_img.squeeze(0),rae_img.squeeze(0)
    ori_np, rae_np = ori_img.cpu().numpy(),rae_img.cpu().numpy()
    ori_np, rae_np = np.transpose(ori_np,(1,2,0)), np.transpose(rae_np,(1,2,0))
    # cal SSIM
    SSIM = skmetrics.structural_similarity(ori_np, rae_np, multichannel=True)
    # cal PSNR
    PSNR = skmetrics.peak_signal_noise_ratio(ori_np, rae_np)
    # cal l2 norm
    l2norm = cv.norm(ori_np, rae_np, cv.NORM_L2)
    # cal linf norm
    linfnorm = cv.norm(ori_np, rae_np, cv.NORM_INF)
    return SSIM,PSNR,l2norm,linfnorm