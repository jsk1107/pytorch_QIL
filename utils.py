# utils.py
import torch
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

        print(f'HOOK 설정 Layer: {self.module_name}-{self.module.name}')
        value = input[0].view(-1).detach().cpu().numpy()
        value = value[value > 0]
        out = output[0].view(-1).detach().cpu().numpy()
        if self.module.name == 'weight':
            self.weight = out
            percentile_90 = np.percentile(value, 90)
            value = value[value < percentile_90]
            hist = np.histogram(value, bins=10)[1][1:]
        elif self.module.name == 'activation':
            self.activation = out
            percentile_90 = np.percentile(value, 90)
            value = value[value < percentile_90]
            hist = np.histogram(value, bins=10)[1][1:]

        prun_point, clip_point = np.min(hist), np.max(hist)
        c_delta = (prun_point + clip_point) / 2
        d_delta = clip_point - c_delta
        module.c_delta.data = torch.nn.Parameter(torch.tensor([c_delta], dtype=torch.float))
        module.d_delta.data = torch.nn.Parameter(torch.tensor([d_delta], dtype=torch.float))

    def graph_prun_ratio(self):
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




def prun_ratio(handle_hooks, plot=False):

    # prun ratio 그래프 그리기
    weight_prun_ratio = {}
    activation_prun_ratio = {}
    for handle_hook in handle_hooks:
        name = handle_hook.module.name
        if name == 'weight':
            inputs = handle_hook.weight
            ratio = np.count_nonzero(inputs == 0.) / len(inputs)

            weight_prun_ratio[handle_hook.module_name] = ratio
        elif name == 'activation':
            inputs = handle_hook.activation
            ratio = np.count_nonzero(inputs == 0.) / len(inputs)
            activation_prun_ratio[handle_hook.module_name] = ratio

    if plot:
        xlim = np.arange(len(activation_prun_ratio))+1
        plt.bar(activation_prun_ratio.keys(), activation_prun_ratio.values())
        plt.xticks(xlim, xlim)
        plt.ylabel('Activation Pruning Ratio')
        plt.show()

        xlim = np.arange(len(weight_prun_ratio))+1
        plt.bar(weight_prun_ratio.keys(), weight_prun_ratio.values())
        # plt.bar(xlim, weight_prun_ratio)
        plt.xticks(xlim, xlim)
        plt.ylabel('Weight Pruning Ratio')
        plt.show()

    return weight_prun_ratio, activation_prun_ratio


def check_interval_param(logger, model):

    for name, module in model.named_modules():
        if isinstance(module, Transformer):
            layer_name = module.name
            if layer_name == 'weight':
                logger.info(
                    f'{name}{layer_name}\t'
                    f'\t c_w \t {round(module.c_delta.data.item(), 5)}'
                    f'\t d_w \t {round(module.d_delta.data.item(), 5)}')
            if layer_name == 'activation':
                logger.info(
                    f'{name}{layer_name}\t'
                    f'\t c_w \t {round(module.c_delta.data.item(), 5)}'
                    f'\t d_w \t {round(module.d_delta.data.item(), 5)}')

