import torch
import torch.nn as nn

# Convert RGB tensor to YUV444
def rgb2yuv2(images):
    yuv_img = torch.zeros_like(images)
    R = images[:, 0, :, :]
    G = images[:, 1, :, :]
    B = images[:, 2, :, :]
    yuv_img[:, 0, :, :] = 0.299 * R + 0.587 * G + 0.114 * B
    yuv_img[:, 1, :, :] = -0.1687 * R - 0.3313 * G + 0.500 * B + 128 / 255
    yuv_img[:, 2, :, :] = 0.500 * R - 0.4187 * G - 0.0813 * B + 128 / 255
    return yuv_img

# Convert YUV444 tensor to RGB
def yuv2rgb2(images):
    rgb_img = torch.zeros_like(images)
    Y = images[:, 0, :, :]
    U = images[:, 1, :, :]
    V = images[:, 2, :, :]
    rgb_img[:, 0, :, :] = Y + 1.402 * (V - 128 / 255)
    rgb_img[:, 1, :, :] = Y - 0.34414 * (U - 128 / 255) - 0.71414 * (V - 128 / 255)
    rgb_img[:, 2, :, :] = Y + 1.772 * (U - 128 / 255)
    return rgb_img

# PGD Attack in YUV space
def atk_PGD(model, device, images, labels, eps=4/255, alpha=1/255, steps=10, channel=0):
    loss = nn.CrossEntropyLoss()
    images, labels = images.to(device), labels.to(device)
    yuv_images = rgb2yuv2(images)
    yuv_images[:, channel, :, :] += torch.empty_like(yuv_images[:, channel, :, :]).uniform_(-eps, eps)
    yuv_images = torch.clamp(yuv_images, min=0, max=1)
    for _ in range(steps):
        yuv_images.requires_grad = True
        re_rgb_image = yuv2rgb2(yuv_images)
        output = model(re_rgb_image)
        cost = loss(output, labels)
        model.zero_grad()
        cost.backward()
        perturbated_yuv = yuv_images.detach().clone()
        perturbated_yuv[:, channel, :, :] += alpha * torch.sign(yuv_images.grad[:, channel, :, :])
        perturbated_rgb = yuv2rgb2(perturbated_yuv)
        delta = torch.clamp(perturbated_rgb - images, min=-eps, max=eps)
        perturbated_rgb = torch.clamp(images + delta, min=0, max=1)
        yuv_images = rgb2yuv2(perturbated_rgb)
    return perturbated_rgb

# BIM Attack in YUV space
def atk_BIM(model, device, images, labels, eps=4/255, alpha=1/255, steps=20, channel=0):
    loss = nn.CrossEntropyLoss()
    images, labels = images.to(device), labels.to(device)
    yuv_images = rgb2yuv2(images)
    yuv_images = torch.clamp(yuv_images, min=0, max=1)
    for _ in range(steps):
        yuv_images.requires_grad = True
        re_rgb_image = yuv2rgb2(yuv_images)
        output = model(re_rgb_image)
        cost = loss(output, labels)
        model.zero_grad()
        cost.backward()
        perturbated_yuv = yuv_images.detach().clone()
        perturbated_yuv[:, channel, :, :] += alpha * torch.sign(yuv_images.grad[:, channel, :, :])
        perturbated_rgb = yuv2rgb2(perturbated_yuv)
        a = torch.clamp(images - eps, min=0)
        b = (perturbated_rgb >= a).float() * perturbated_rgb + (perturbated_rgb < a).float() * a
        c = (b > images + eps).float() * (images + eps) + (b <= images + eps).float() * b
        perturbated_rgb = torch.clamp(c, max=1)
        yuv_images = rgb2yuv2(perturbated_rgb)
    return perturbated_rgb

# FGSM Attack in YUV space
def atk_FGSM(model, device, images, labels, eps=4/255, channel=0):
    loss = nn.CrossEntropyLoss()
    images, labels = images.to(device), labels.to(device)
    yuv_images = rgb2yuv2(images)
    yuv_images = torch.clamp(yuv_images, min=0, max=1)
    yuv_images.requires_grad = True
    re_rgb_image = yuv2rgb2(yuv_images)
    output = model(re_rgb_image)
    cost = loss(output, labels)
    model.zero_grad()
    cost.backward()
    perturbated_yuv = yuv_images.detach().clone()
    perturbated_yuv[:, channel, :, :] += eps * torch.sign(yuv_images.grad[:, channel, :, :])
    perturbated_rgb = yuv2rgb2(perturbated_yuv)
    delta = torch.clamp(perturbated_rgb - images, min=-eps, max=eps)
    perturbated_rgb = torch.clamp(images + delta, min=0, max=1)
    return perturbated_rgb

# MI-FGSM (Momentum Iterative) Attack in YUV space
def atk_MI_YFGSM(model, device, images, labels, eps=4/255, alpha=1/255, steps=20, mu=1.0, channel=0):
    loss = nn.CrossEntropyLoss()
    images, labels = images.to(device), labels.to(device)
    yuv_images = rgb2yuv2(images)
    g = torch.zeros_like(yuv_images).detach().to(device)
    yuv_images = torch.clamp(yuv_images, min=0, max=1)
    for _ in range(steps):
        yuv_images.requires_grad = True
        re_rgb_image = yuv2rgb2(yuv_images)
        output = model(re_rgb_image)
        cost = loss(output, labels)
        model.zero_grad()
        cost.backward()
        grad = yuv_images.grad[:, channel, :, :]
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2), keepdim=True) + 1e-8
        normalized_grad = grad / grad_norm
        g[:, channel, :, :] = mu * g[:, channel, :, :] + normalized_grad
        perturbated_yuv = yuv_images.detach().clone()
        perturbated_yuv[:, channel, :, :] += alpha * torch.sign(g[:, channel, :, :])
        perturbated_rgb = yuv2rgb2(perturbated_yuv)
        delta = torch.clamp(perturbated_rgb - images, min=-eps, max=eps)
        perturbated_rgb = torch.clamp(images + delta, min=0, max=1)
        yuv_images = rgb2yuv2(perturbated_rgb).detach()
    return perturbated_rgb

# NI-FGSM (Nesterov Iterative) Attack in YUV space
def atk_NI_YFGSM(model, device, images, labels, eps=4/255, alpha=1/255, steps=20, mu=1.0, channel=0):
    loss = nn.CrossEntropyLoss()
    images, labels = images.to(device), labels.to(device)
    yuv_images = rgb2yuv2(images)
    g = torch.zeros_like(yuv_images).detach().to(device)
    yuv_images = torch.clamp(yuv_images, min=0, max=1)
    for _ in range(steps):
        yuv_images.requires_grad = True
        nesterov_yuv = yuv_images.clone()
        nesterov_yuv[:, channel, :, :] = yuv_images[:, channel, :, :] + mu * alpha * g[:, channel, :, :]
        nesterov_yuv = torch.clamp(nesterov_yuv, min=0, max=1)
        re_rgb_image = yuv2rgb2(nesterov_yuv)
        output = model(re_rgb_image)
        cost = loss(output, labels)
        model.zero_grad()
        cost.backward()
        grad = yuv_images.grad[:, channel, :, :]
        grad_norm = torch.mean(torch.abs(grad), dim=(1, 2), keepdim=True) + 1e-8
        normalized_grad = grad / grad_norm
        g[:, channel, :, :] = mu * g[:, channel, :, :] + normalized_grad
        perturbated_yuv = yuv_images.detach().clone()
        perturbated_yuv[:, channel, :, :] += alpha * torch.sign(g[:, channel, :, :])
        perturbated_rgb = yuv2rgb2(perturbated_yuv)
        delta = torch.clamp(perturbated_rgb - images, min=-eps, max=eps)
        perturbated_rgb = torch.clamp(images + delta, min=0, max=1)
        yuv_images = rgb2yuv2(perturbated_rgb).detach()
    return perturbated_rgb
