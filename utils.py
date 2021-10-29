# utils.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

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