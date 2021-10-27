import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Hook:

    def __init__(self, module, module_name, backward=False):
        self.module_name = module_name
        if backward:
            self.hook = module.register_backward_hook(self.hook_fn)
        else:
            self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):

        if module.name == 'activation':
            value = input[0].view(-1).detach().cpu().numpy()
            plt.hist(value, bins=25)
            plt.xlabel(f'{self.module_name} input')
            plt.ylabel('count')
            plt.show()
            #
            # value = output[0].view(-1).detach().cpu().numpy()
            # plt.hist(value)
            # plt.xlabel(f'{self.module_name} output')
            # plt.ylabel('count')
            # plt.show()

    def remove(self):
        self.hook.remove()

    # if isinstance(module, Conv2d):
    #     for name_1, param in module.named_parameters():
    #         if name_1 == 'weight':
    #             tmp = param.view(-1).detach().cpu().numpy()
    #             plt.hist(tmp)
    #             plt.xlabel(f'{name} {name_1}')
    #             plt.ylabel('count')
    #             plt.show()