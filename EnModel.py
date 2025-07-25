from torchvision import models, transforms
from torch import nn


class EnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_Dense = models.densenet161(pretrained=True)
        self.model_Inc = models.inception_v3(pretrained=True)
        self.model_Goo = models.googlenet(pretrained=True)
        self.num_of_model = 3

    def trans_256224(self, input):
        trans = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        return trans(input)

    def trans_299(self, input):
        trans = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299)
        ])
        return trans(input)

    def forward(self, input):
        output1 = self.model_Dense(input)
        output2 = self.model_Inc(input)
        output3 = self.model_Goo(input)
        return 1 / 3 * output1 + 1 / 3 * output2 + 1 / 3 * output3
