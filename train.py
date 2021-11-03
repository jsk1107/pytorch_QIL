import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageNet, ImageFolder
from torchvision.transforms import transforms as T
from resnet import resnet20, resnet18
# from torchvision.models.resnet import resnet18
import torch.optim as optim
from tqdm import tqdm
from utils import Hook
import numpy as np
import matplotlib.pyplot as plt
from qil import Transformer
from logger import get_logger


""" Logger 등록 """
logger = get_logger('./log', log_config='./logging.json')


""" Data Setting """
def custom_transfroms_cifar10(train=True):

    if train:
        return T.Compose([ T.RandomCrop(32, padding=4),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                           ])
    else:
        return T.Compose([ T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                           ])

def custom_transfroms_imagenet(train=True):

    if train:
        return T.Compose([ T.Resize(size=(256, 256)),
                           T.RandomCrop(224, padding=4),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                           ])
    else:
        return T.Compose([ T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                           ])

# train_set = CIFAR10('../data', train=True, transform=custom_transfroms_cifar10(True), download=True)
# val_set = CIFAR10('../data', train=False, transform=custom_transfroms_cifar10(False))

train_set = ImageFolder('../data/imagenet/', transform=custom_transfroms_imagenet(True))
val_set = ImageFolder('../data/imagenet/', transform=custom_transfroms_imagenet(False))

train_loader = DataLoader(train_set, batch_size=256, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=256, shuffle=False, pin_memory=True)

""" CONFIG """

EPOCHS = 120

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


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


def main(**kwargs):
    pretrain_model_path = 'final_fp32_model.pt'
    # model = resnet20(pretrain_model_path, **kwargs)
    model = resnet18(pretrain_model_path, **kwargs)
    # model = resnet18(False)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': interval_param_list(model), 'lr': 1e-4},
                           {'params': weight_param_list(model), 'lr': 4e-2}], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 85, 95, 105], gamma=0.1)
    model.to(device)

    # # Hook 등록
    # handle_hooks = []
    # for module_name, module in model.named_modules():
    #     if isinstance(module, Transformer):
    #         # if module.name == 'activation':
    #         h = Hook(module, module_name)
    #         handle_hooks.append(h)
    #
    # # 1024장 이미지를 통해 c_x, d_x 초기값 정하기
    # model.train()
    # tmp_dataloader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
    # for data in tmp_dataloader:
    #     optimizer.zero_grad()
    #
    #     imgs, targets = data
    #     imgs, targets = imgs.to(device), targets.to(device)
    #
    #     output = model(imgs)
    #     iter_loss = criterion(output, targets)
    #     iter_loss.backward()
    #     print('c_delta, d_delta 초기값 셋팅 완료')
    #     break
    #
    # # Interval Param Check
    # check_interval_param(model)
    #
    # # Prun_ratio 그래프 그리기
    # graph_prun_ratio(handle_hooks, plot=False)
    #
    # # Hook 제거
    # for handle_hook in handle_hooks:
    #     handle_hook.handle.remove()

    best_acc = .0
    for EPOCH in range(EPOCHS):

        model.train()
        train_loss = .0

        with tqdm(train_loader) as tbar:
            for i, data in enumerate(tbar):
                optimizer.zero_grad()

                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)

                output = model(imgs)
                iter_loss = criterion(output, targets)
                train_loss += iter_loss

                iter_loss.backward()
                optimizer.step()

                tbar.set_description(
                    f'EPOCH: {EPOCH + 1} | total_train_loss: {train_loss / (i+1):.4f} | batch_loss: {iter_loss:.4f}'
                )

            scheduler.step(EPOCH)

        model.eval()
        val_loss = .0
        correct = .0
        total = 0

        with tqdm(val_loader) as tbar:
            for i, data in enumerate(tbar):

                imgs, targets = data
                imgs, targets = imgs.to(device), targets.to(device)

                with torch.no_grad():
                    output = model(imgs)
                iter_loss = criterion(output, targets)
                val_loss += iter_loss

                tbar.set_description(
                    f'EPOCH: {EPOCH + 1} | total_val_loss: {val_loss / (i + 1):.4f} | batch_loss: {iter_loss:.4f}'
                )

                _, pred_idx = torch.max(output, dim=1)
                total += targets.size(0)
                correct += (pred_idx == targets).sum()

        acc = 100 * correct / total

        # Interval Param Check
        check_interval_param(model)

        if best_acc < acc:
            best_acc = acc
            check_point = {'state_dict': model.state_dict(),
                           'optim': optimizer.state_dict(),
                           'EPOCH': EPOCH,
                           'Acc': best_acc.round()}
            torch.save(check_point, f'./model_{kwargs.get("w_bit")}_{kwargs.get("a_bit")}_model.pt')

        logger.info(f'EPOCH: {EPOCH + 1} | '
                    f'Loss: {val_loss / (i + 1):.4f} | '
                    f'Accuracy: {acc:.4f}%')
    logger.info(f'Best_Acc: {best_acc:.4f}%')


if __name__ == '__main__':

    # bits = [[32, 32], [5, 5], [4, 4], [3, 3], [2, 2]]
    bits = [[32, 32]]
    for i in bits:
        kwargs = {'w_bit': i[0], 'a_bit': i[1]}
        main(**kwargs)