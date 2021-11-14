import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, name):
        super(Transformer, self).__init__()

        self.name = name

        """ Transformer Parameter """
        self.c_delta = nn.Parameter(torch.tensor([0.5], device='cuda'))
        self.d_delta = nn.Parameter(torch.tensor([0.5], device='cuda'))
        if name == 'weight':
            self.gamma = torch.tensor([1.0], device='cuda')
        else:
            self.gamma = None

    def forward(self, x):

        self.c_delta.data = torch.abs(self.c_delta)
        self.d_delta.data = torch.abs(self.d_delta)

        # 0으로 나눠지게 되는것 방지. (nan, inf 발생)
        if self.d_delta.data < 0.001:
            self.d_delta.data += 1e-10

        prun_point = self.c_delta - self.d_delta
        clip_point = self.c_delta + self.d_delta

        alpha = 0.5 / self.d_delta
        beta = ((- 0.5 * self.c_delta) / self.d_delta) + 0.5

        if self.name == 'weight':
            return self._weight_transform(x, alpha, beta, prun_point, clip_point)
        elif self.name == 'activation':
            return self._activation_transform(x, alpha, beta, prun_point, clip_point)
        else:
            raise NotImplementedError()

    def _weight_transform(self, weight, alpha, beta, prun_point, clip_point):

        """ transformer T_w Eq(3) """
        hat_w = torch.where(torch.lt(torch.abs(weight), prun_point), torch.tensor(0.0, device=0),
                           torch.where(torch.gt(torch.abs(weight), clip_point), torch.sign(weight),
                                       torch.pow(alpha * torch.abs(weight) + beta,
                                                 self.gamma) * torch.sign(weight)))

        return hat_w

    def _activation_transform(self, x, alpha, beta, prun_point, clip_point):

        """ transformer T_x Eq(5) """
        hat_x = torch.where(torch.lt(x, prun_point), torch.tensor(0.0, device=0),
                            torch.where(torch.gt(x, clip_point), torch.tensor(1.0, device=0),
                                        alpha * x + beta))

        return hat_x


# class Transformer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, c_delta, d_delta, gamma, name):
#         #
#         # c_delta = torch.where(c_delta <= 0, torch.tensor(0.5, device=0), c_delta)
#         # d_delta = torch.where(d_delta <= 0, torch.tensor(1.0, device=0), d_delta)
#         if c_delta < 0 or d_delta < 0:
#             c_delta = torch.abs(c_delta)
#             d_delta = torch.abs(d_delta)
#
#
#         alpha = 0.5 / d_delta
#         beta = ((- 0.5 * c_delta) / d_delta) + 0.5
#
#         prun_point = c_delta - d_delta
#         clip_point = c_delta + d_delta
#         prun_point = torch.where(prun_point <= 0, torch.tensor([0.0], device=0), prun_point)
#         clip_point = torch.where(clip_point >= 1, torch.tensor([1.0], device=0), clip_point)
#
#         if name == 'weight':
#             mask = (torch.abs(x) > prun_point) * (torch.abs(x) < clip_point) # Interval
#             tmp_weight = (torch.pow((alpha * torch.abs(x)) + beta, gamma) * torch.sign(x)) * mask
#             tmp_weight_2 = tmp_weight + (torch.sign(x) * (torch.abs(x) > clip_point))
#             ctx.save_for_backward(x, mask, alpha, beta, c_delta, d_delta, gamma, torch.Tensor([True]))
#             return tmp_weight_2
#         elif name == 'activation':
#             mask = (x > prun_point) * (x < clip_point) # Interval
#             tmp_x = (alpha * x + beta) * mask
#             tmp_x_2 = tmp_x + (x > clip_point)
#             ctx.save_for_backward(x, mask, alpha, beta, c_delta, d_delta, gamma, torch.Tensor([False]))
#             return tmp_x_2
#         else:
#             raise NotImplementedError
#
#     @staticmethod
#     def backward(ctx, grad_outputs):
#         x, mask, alpha, beta, c_delta, d_delta, gamma, flag = ctx.saved_tensors
#         grad_input = grad_c_delta = grad_d_delta = grad_gamma = name = None
#         if flag == 1:
#             # 공통 미분값
#             common = (gamma * (alpha * torch.abs(x) + beta) ** (gamma-1)) * torch.sign(x)
#
#             grad_input = (common * alpha) * grad_outputs * mask # weight 편미분
#             grad_c_delta = (-alpha * common) * grad_outputs * mask # c_delta 편미분
#             grad_d_delta = (common * (alpha / d_delta) * (c_delta - torch.abs(x))) * grad_outputs * mask # d_delta 편미분
#
#             # print('grad_input', grad_input)
#             # print('grad_c_delta', grad_c_delta.sum())
#             # print('grad_d_delta', grad_d_delta)
#             return grad_input, grad_c_delta.sum(0), grad_d_delta.sum(0), None, None
#         elif flag == 0:
#
#             grad_input = alpha * grad_outputs * mask # x 편미분
#             grad_c_delta = - alpha * grad_outputs * mask # c_delta 편미분
#             grad_d_delta = (alpha / d_delta) * (c_delta - x) * grad_outputs * mask # d_delta 편미분
#
#             return grad_input, grad_c_delta.sum(0), grad_d_delta.sum(0), None, None
#         else:
#             raise NotImplementedError()


class Discretizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hat_w, bitw, name):
        ctx.save_for_backward(hat_w)

        """
            Paper에서 DoReFaNet STE를 사용한다고 밝히고 있다.
            We use straight-through-estimator [2, 27] for the gradient of the discretizers.
        """

        """ Discretizer D_w Eq(2) """
        if name == 'weight':
            quant_level = 2 ** (bitw - 1) - 1
            return 2 * (torch.round((0.5 * hat_w + 0.5) * quant_level) / quant_level) - 1
        elif name == 'activation':
            quant_level = (2 ** bitw) - 1
            return torch.round(hat_w * quant_level) / quant_level

    @staticmethod
    def backward(ctx, grad_outputs):
        weight, = ctx.saved_tensors
        gate = (torch.abs(weight) <= 1).float()
        grad_inputs = grad_outputs * gate
        return grad_inputs, None, None


class Quantizer(nn.Module):
    def __init__(self, bit, name):
        super(Quantizer, self).__init__()
        self.bit = bit
        self.name = name
        self.transformer = Transformer(name)

    def forward(self, x):

        if self.bit == 32:
            return x
        transform_x = self.transformer(x)
        quantized_x = Discretizer.apply(transform_x, self.bit, self.name)

        return quantized_x


class QConv2d(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, bit=32):
        super(QConv2d, self).__init__(
            in_channel, out_channel, kernel_size, stride, padding, dilation, groups, bias)
        self.bit = bit
        self.quantized_weight = Quantizer(bit, 'weight')

    def forward(self, x):
        w_q = self.quantized_weight(self.weight)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QActivation(nn.Module):
    def __init__(self, bit=32):
        super(QActivation, self).__init__()
        self.quantized_activation = Quantizer(bit, 'activation')

    def forward(self, x):
        return self.quantized_activation(x)


if __name__ == '__main__':

    import torch
    import random
    import matplotlib.pyplot as plt

    seed_number = 1
    torch.manual_seed(seed_number)
    random.seed(seed_number)
    transformer = Transformer(name='activation')
    bit = 3
    inputs = torch.normal(0, 1.5, (3, 3, 28, 28), device=0)

    conv2d = Quant_Conv2d(3, 32, 3, bit=bit).to('cuda')
    activation = Quant_Activation(bit=bit).to('cuda')
    relu = nn.ReLU()

    out = conv2d(inputs)
    out1 = relu(out)
    out2 = activation(out1)

    x = out1.view(-1).detach().cpu().numpy()
    q_x = out2.view(-1).detach().cpu().numpy()

    w = conv2d.weight
    w_hat = conv2d.quantized_weight(conv2d.weight)

    w = w.view(-1).detach().cpu().numpy()
    w_hat = w_hat.view(-1).detach().cpu().numpy()

    plt.plot(x, q_x, 'ro')
    plt.axis([0, 1.5, -1, 1])
    plt.show()

    plt.plot(w, w_hat, 'ro')
    plt.axis([-0.2, 0.2, -1, 1])
    plt.show()

    plt.hist(w)
    plt.ylabel('count')
    plt.xlabel('hat_w')
    plt.show()