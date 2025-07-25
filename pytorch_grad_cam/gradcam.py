import cv2
import numpy as np
import torch
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

class CAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def gradcampp(self, activations, grads):
        grads_power_2 = grads**2
        grads_power_3 = grads_power_2*grads
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 0.00000001
        aij = grads_power_2 / (2*grads_power_2 + sum_activations[:, None, None]*grads_power_3 + eps)

        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0)*aij
        weights = np.sum(weights, axis=(1, 2))
        return weights

    def scorecam(self, 
                 input_tensor, 
                 activations, 
                 target_category,
                 original_score):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(size=input_tensor.shape[2 : ])
            activation_tensor = torch.from_numpy(activations).unsqueeze(0)
            if self.cuda:
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor)
            upsampled = upsampled[0, ]
            
            maxs = upsampled.view(upsampled.size(0), -1).max(dim=-1)[0]
            mins = upsampled.view(upsampled.size(0), -1).min(dim=-1)[0]
            maxs, mins = maxs[:, None, None], mins[:, None, None]
            upsampled = (upsampled - mins) / (maxs - mins)

            input_tensors = input_tensor*upsampled[:, None, :, :]
            batch_size = 16
            scores = []
            for i in range(0, input_tensors.size(0), batch_size):
                batch = input_tensors[i : i + batch_size, :]
                outputs = self.model(batch).cpu().numpy()[:, target_category]
                scores.append(outputs)
            scores = torch.from_numpy(np.concatenate(scores))
            weights = torch.nn.Softmax(dim=-1)(scores - original_score).numpy()
            return weights

    # __call__可以让实例对象可以像函数一样被调用
    def __call__(self, input_tensor, method="gradcam", target_category=None):
        # 将输入图像tensor移到GPU
        if self.cuda:
            input_tensor = input_tensor.cuda()
        # output是self.model对input_tensor的输出，并通过前向hook得到self.target_layer的特征图，创建反向hook为计算梯度作准备
        output = self.activations_and_grads(input_tensor)
        # 找出input_tensor的类别
        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy())
        # 构建input_tensor的类别所对应的one-hot向量
        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        # 将one-hot加入计算图，并求关于特征图的梯度
        one_hot = torch.sum(one_hot * output)
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        # activations存储特征图 [0,:]用来降维
        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        # grads存储特征图梯度
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        if method == "gradcam++":
            weights = self.gradcampp(activations, grads)
        elif method == "gradcam":
            # gradcam:将每个特征图梯度求element-eise平均得到各特征图的权重
            weights = np.mean(grads, axis=(1, 2))
        elif method == "scorecam":
            original_score = original_score=output[0, target_category].cpu()
            weights = self.scorecam(input_tensor, 
                                    activations, 
                                    target_category,
                                    original_score=original_score)
        else:
            raise "Method not supported"
        # 各特征图叠加得到cam(共2048张10*10的特征图)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        # 去掉cam中的负值
        cam = np.maximum(cam, 0)
        # 经过上面这步后注意力图还是和特征图一样大(10*10)
        # 接下来对注意力图进行缩放，使其达到299*299
        # [::-1]可用来翻转列表等
        cam = cv2.resize(cam, input_tensor.shape[2:][::-1])
        # 将cam线性归一化
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam
