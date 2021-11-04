# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from qil import Transformer


class Hook:

    def __init__(self, module, module_name=None, backward=False):
        self.module = module
        self.module_name = module_name
        self.weight = None
        self.activation = None

        if backward:
            self.handle = module.register_backward_hook(self.hook_fn)
        else:
            self.handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        print('------------', self.module_name, '.', self.module.name, '------------')
        value = input[0].view(-1).detach().cpu().numpy()
        value = value[value > 0]
        out = output[0].view(-1).detach().cpu().numpy()
        if self.module.name == 'weight':
            self.weight = out
            percentile_90 = np.percentile(value, 90)
            hist = np.histogram(value, bins=50)[1][1:]
            hist = hist[hist < percentile_90]
        elif self.module.name == 'activation':
            self.activation = out
            percentile_90 = np.percentile(value, 90)
            hist = np.histogram(value, bins=50)[1][1:]
            hist = hist[hist < percentile_90]


        prun_point, clip_point = np.min(hist), np.max(hist)
        c_delta = (prun_point + clip_point) / 2
        d_delta = clip_point - c_delta
        module.c_delta.data = torch.nn.Parameter(torch.tensor([c_delta], device=0, dtype=torch.float))
        module.d_delta.data = torch.nn.Parameter(torch.tensor([d_delta], device=0, dtype=torch.float))

        # plt.hist(value, bins=30)
        # plt.xlabel(f'{self.module_name} input')
        # plt.ylabel('count')
        # plt.show()

    def graph_prun_ratio(self):
        # x_label = ''.join(self.module_name.split('.')[:3])

        prun_ratio = []
        for idx, value in enumerate(self.input):
            prun_ratio.append((len(value) - np.count_nonzero(value)) / len(value))

        plt.bar(self.input, prun_ratio)
        plt.xlabel(f'{idx}')
        plt.ylabel('Pruning Ratio')
        plt.show()


def interval_param_list(model):
    for name, param in model.named_parameters():
        target = name.split('.')[-1].split('_')[-1]
        if target == 'delta':
            yield param


def weight_param_list(model):
    for name, param in model.named_parameters():
        target = name.split('.')[-1].split('_')[-1]
        if target != 'delta':
            yield param



def graph_prun_ratio(handle_hooks, plot=False):
    if not plot:
        return

    # prun ratio 그래프 그리기
    activation_prun_ratio = []
    weight_prun_ratio = []
    for handle_hook in handle_hooks:
        name = handle_hook.module.name
        if name == 'weight':
            inputs = handle_hook.weight
            ratio = (len(inputs) - np.count_nonzero(inputs)) / len(inputs)
            weight_prun_ratio.append(ratio)
        elif name == 'activation':
            inputs = handle_hook.activation
            ratio = (len(inputs) - np.count_nonzero(inputs)) / len(inputs)
            activation_prun_ratio.append(ratio)

    xlim = np.arange(len(activation_prun_ratio))+1
    plt.bar(xlim, activation_prun_ratio)
    plt.xticks(xlim, xlim)
    plt.ylabel('Activation Pruning Ratio')
    plt.show()

    xlim = np.arange(len(weight_prun_ratio))+1
    plt.bar(xlim, weight_prun_ratio)
    plt.xticks(xlim, xlim)
    plt.ylabel('Weight Pruning Ratio')
    plt.show()


def check_interval_param(model):

    for name, module in model.named_modules():
        if isinstance(module, Transformer):
            layer_name = module.name
            if layer_name == 'weight':
                print(
                    f'{name}{layer_name} ||'
                    f' c_w: {round(module.c_delta.data.item(), 5)},'
                    f' d_w: {round(module.d_delta.data.item(), 5)}')
            if layer_name == 'activation':
                print(
                    f'{name}{layer_name} ||'
                    f' c_x: {round(module.c_delta.data.item(), 5)},'
                    f' d_x: {round(module.d_delta.data.item(), 5)}')

