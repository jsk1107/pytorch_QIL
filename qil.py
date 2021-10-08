import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, name):
        super(Transformer, self).__init__()

        self.name = name

        """ Transformer Parameter """
        self.c_w = nn.Parameter(torch.tensor([0.2]))
        self.d_w = nn.Parameter(torch.tensor([0.2]))

        self.alpha_w = None
        self.beta_w = None

        # TODO: gamma를 고정할것인지 학습할것인지는 선택 가능.
        self.gamma = nn.Parameter(torch.Tensor([5.0]))

    def forward(self, x):

        self.alpha = 0.5 / self.d_w
        self.beta = -0.5 * self.c_w / self.d_w + 0.5

        """ transformer T_w Eq(3) """
        if self.name == 'weight':
            return self._weight_transform(x)
        elif self.name == 'activation':
            return self._activation_transform(x)
        else:
            raise NotImplementedError()

    def _weight_transform(self, weight):

        hat_w = torch.where(torch.lt(torch.abs(weight), (self.c_w - self.d_w)), torch.tensor(0.0, device=0),
                            torch.where(torch.gt(torch.abs(weight), (self.c_w + self.d_w)), torch.sign(weight),
                                        torch.pow(self.alpha * torch.abs(weight) + self.beta,
                                                  self.gamma) * torch.sign(weight)))
        return hat_w

    def _activation_transform(self, x):

        hat_x = torch.where(torch.lt(x, self.c_w - self.d_w), torch.tensor(0.0, device=0),
                            torch.where(torch.gt(x, (self.c_w + self.d_w)), torch.tensor(1.0, device=0),
                                        self.alpha * x + self.beta))
        return hat_x


class Discretizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hat_w, bitw):

        """ Discretizer D_w Eq(2) """
        quant_level = 2 ** (bitw - 1) - 1
        bar_w = torch.round(hat_w * quant_level) / quant_level
        return bar_w

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class Quantizer(nn.Module):
    def __init__(self, bit, name):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.transformer = Transformer(name)

    def forward(self, x):

        transform_x = self.transformer(x)
        quantized_x = Discretizer.apply(transform_x, self.bit)

        return quantized_x


class Quant_Conv2d(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, bits=32):
        super(Quant_Conv2d, self).__init__(
            in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias)
        self.quantized_weight = Quantizer(bits, 'weight')

    def forward(self, x):

        w_q = self.quantized_weight(self.weight)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Quant_Activation(nn.Module):
    def __init__(self, bits):
        super(Quant_Activation, self).__init__()
        self.quantized_activation = Quantizer(bits, 'activation')

    def forward(self, x):
        return self.quantized_activation(x)


if __name__ == '__main__':

    import torch
    import random
    import matplotlib.pyplot as plt
    from torchvision.models import resnet

    seed_number = 1
    torch.manual_seed(seed_number)
    random.seed(seed_number)
    transformer = Transformer(name='activation')
    bits = 3
    inputs = torch.normal(0, 1.5, (3, 3, 28, 28))

    conv2d = Quant_Conv2d(3, 32, 3, bits=bits)
    activation = Quant_Activation(bits=bits)
    relu = nn.ReLU()


    out = conv2d(inputs)
    out1 = relu(out)
    out2 = activation(out1)

    x = out1.view(-1).detach().numpy()
    q_x = out2.view(-1).detach().numpy()
    # weight = weight.view(-1).detach().numpy()
    # quantizer = quantizer.view(-1).detach().numpy()

    print('x',x )
    print('q_x', q_x)
    plt.plot(x, q_x, 'ro')
    plt.axis([-0.4, 1, -1, 1])
    plt.show()