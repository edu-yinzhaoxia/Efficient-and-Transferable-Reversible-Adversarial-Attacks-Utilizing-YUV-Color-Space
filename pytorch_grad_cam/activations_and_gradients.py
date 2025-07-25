class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model

        self.gradients = []
        # 储存target_layer在前向传播中得到的特征图
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    # 前向钩子，hook得到target_layer在前向传播中得到的特征图
    def save_activation(self, module, input, output):
        # print(input[0].shape)
        # print(output[0].shape)
        self.activations.append(output)

    # 反向钩子
    def save_gradient(self, module, grad_input, grad_output):
        # print(grad_input[0].shape)
        # print(grad_output[0].shape)
        # Gradients are computed in reverse order
        self.gradients = [grad_input[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []        
        return self.model(x)