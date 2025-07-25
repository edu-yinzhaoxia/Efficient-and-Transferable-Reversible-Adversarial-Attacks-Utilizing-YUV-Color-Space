import torch
from torchvision import models
import torchvision.transforms as transforms
from utils import rgb2yuv, yuv2rgb
from embed_utils import embed_main
import numpy as np
from torchvision import utils as Vutils
from PIL import Image
import random
from pytorch_grad_cam import CAM
from atk import atk_MI_YFGSM
from calMetrics import calMetrics
import os
from EnModel import EnModel

def seed_everything(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(2022)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load classification model
model = models.inception_v3(pretrained=True).to(device)
model.eval()

# Load ensemble model
ens_model = EnModel().to(device)
ens_model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
])

transform_ToTensor = transforms.Compose([
    transforms.ToTensor()
])

folder_path = "./ORI_IMG/"

# Attention model for CAM
model1 = models.resnet50(pretrained=True).to(device)
target_layer = model1.layer4[-1]
cam = CAM(model=model1, target_layer=target_layer, use_cuda=True)

def cam_map(image):
    target_category = None
    grayscale_cam = cam(input_tensor=image,
                        method='gradcam++',
                        target_category=target_category)
    return grayscale_cam

all_count = 0
adv_y_fault = 0

SSIM = PSNR = L2 = LINF = 0

for filename in os.listdir(folder_path):
    all_count += 1
    img_path = os.path.join(folder_path, filename)
    images = Image.open(img_path).convert('RGB')
    images = transform_ToTensor(images).unsqueeze(0).to(device)
    _, labels = filename.split('_')
    labels = labels.split('.')[0]
    labels = torch.tensor([int(labels)]).to(device)

    # Generate CAM attention mask
    ori_map = cam_map(images)
    hight, width = ori_map.shape
    for i in range(hight):
        for j in range(width):
            ori_map[i][j] = 1 if ori_map[i][j] > 0.5 else 0

    ori_yuv = rgb2yuv(images)

    for max_iteration in range(50):
        # Generate adversarial example using MI-YFGSM in ensemble
        adv_images = atk_MI_YFGSM(
            ens_model, eps=2/255, alpha=1/255, steps=20,
            device=device, images=images, labels=labels
        )
        adv_yuv = rgb2yuv(adv_images)

        # Only modify Y channel according to attention mask
        advy_yuv = ori_yuv.copy()
        for i in range(299):
            for j in range(299):
                if ori_map[i][j] == 1:
                    advy_yuv[:, :, 0][i][j] = adv_yuv[:, :, 0][i][j]

        # Embed Y-channel perturbation into UV channels
        reversible_yuv = embed_main(ori_yuv, advy_yuv)
        reversible_u = reversible_yuv[:299, :]
        reversible_v = reversible_yuv[299:, :]
        rae_yuv = advy_yuv.copy()
        rae_yuv[:, :, 1] = reversible_u
        rae_yuv[:, :, 2] = reversible_v

        reversible_ndarray = yuv2rgb(rae_yuv)
        reversible_image = Image.fromarray(np.uint8(reversible_ndarray))

        # Classify RAE to check if attack is successful
        reversible_tensor = transform(reversible_image).unsqueeze(0).to(device)
        outputs = model(reversible_tensor)
        _, pre = torch.max(outputs.data, 1)
        if pre != labels:
            adv_y_fault += 1
            ssim, psnr, l2, linf = calMetrics(images, reversible_tensor)
            SSIM += ssim
            PSNR += psnr
            L2 += l2
            LINF += linf
            break
        else:
            advy_ndarray = yuv2rgb(advy_yuv)
            advy_image = Image.fromarray(np.uint8(advy_ndarray))
            advy_tensor = transform(advy_image).unsqueeze(0).to(device)
            images = advy_tensor

    print('all_count, max_iteration', all_count, max_iteration)
    rae_path = './output/rae/' + f"{all_count:04d}" + '_' + str(labels.item()) + '.png'
    Vutils.save_image(reversible_tensor, rae_path)

print('adv_y_fault, all_count', adv_y_fault, all_count)
