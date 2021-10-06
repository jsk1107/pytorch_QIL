import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()

        """ Transformer Parameter """
        self.c_w = nn.Parameter(torch.tensor([0.4]))
        self.d_w = nn.Parameter(torch.tensor([0.2]))

        self.alpha_w = None
        self.beta_w = None

        # TODO: gamma를 고정할것인지 학습할것인지는 선택 가능.
        self.gamma = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, weight):

        self.alpha_w = 0.5 / self.d_w
        self.beta_w = -0.5 * self.c_w / self.d_w + 0.5

        """ transformer T_w Eq(3) """
        hat_w = torch.where(torch.abs(weight) < self.c_w - self.d_w, torch.tensor(0.0),
                            torch.where(torch.abs(weight) > self.c_w + self.d_w, torch.sign(weight),
                            torch.pow(self.alpha_w * torch.abs(weight) + self.beta_w, self.gamma) * torch.sign(weight)))
        return hat_w


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


class Quantizer(torch.autograd.Function):
    pass


class Quant_Conv2d(nn.Conv2d):
    pass


class Quant_Activation(nn.Module):
    pass


if __name__ == '__main__':

    import torch
    import random
    import matplotlib.pyplot as plt
    from torchvision.models import resnet

    seed_number = 1
    torch.manual_seed(seed_number)
    random.seed(seed_number)
    transformer = Transformer()
    bits = 3
    weight = torch.normal(0, 1.5, (1, 2000))

    print('weight', weight)
    t_w = transformer(weight)
    print('t_w', t_w)
    d_w = Discretizer.apply(t_w, bits)
    print('d_w', d_w)


    plt.plot(weight.detach().numpy()[0], t_w.detach().numpy()[0], 'ro', color='blue')
    plt.plot(weight.detach().numpy()[0], d_w.detach().numpy()[0], 'ro')
    plt.axis([0.2, 0.7, -1, 1])
    plt.show()