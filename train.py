import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms as T
from resnet import resnet20
import torch.optim as optim
from tqdm import tqdm


""" Data Setting """

def custom_transfroms(train=True):

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

DATA_PATH = 'C:\\Users\\QQQ\\PycharmProjects\\dorefanet-pytorch\\data'
train_set = CIFAR10(DATA_PATH, train=True, transform=custom_transfroms(True))
val_set = CIFAR10(DATA_PATH, train=False, transform=custom_transfroms(False))

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=128, shuffle=False, pin_memory=True)

""" CONFIG """

EPOCHS = 200

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


kwarg = {'w_bit': 4, 'a_bit': 4}
model = resnet20(**kwarg)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD([{'params': interval_param_list(model), 'lr': 4e-4},
                       {'params': weight_param_list(model), 'lr': 4e-2}], momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=0.1)
model.to(device)

model_path = 'fp32_model_64epoch.pt'
if model_path:
    print(f'load: {kwarg} | {model_path}')

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)


from qil import Transformer
from torch.nn import Conv2d
import matplotlib.pyplot as plt
for name, module in model.named_modules():

    # if isinstance(module, Conv2d):
    #     for name_1, param in module.named_parameters():
    #         if name_1 == 'weight':
    #             tmp = param.view(-1).detach().cpu().numpy()
    #             plt.hist(tmp)
    #             plt.xlabel(f'{name} {name_1}')
    #             plt.ylabel('count')
    #             plt.show()

    if isinstance(module, Transformer):
        layer_name = module.name
        if layer_name == 'weight':

            module.c_delta.data = torch.nn.Parameter(torch.tensor([0.2], device='cuda'))
            module.d_delta.data = torch.nn.Parameter(torch.tensor([0.15], device='cuda'))
        if layer_name == 'activation':
            module.c_delta.data = torch.nn.Parameter(torch.tensor([0.5], device='cuda'))
            module.d_delta.data = torch.nn.Parameter(torch.tensor([0.45], device='cuda'))

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
    print(f'Acc: {acc:.4f}%')

    for name, module in model.named_modules():
        if isinstance(module, Transformer):
            layer_name = module.name
            if layer_name == 'weight':
                print(f'{name}{layer_name} || c_w: {round(module.c_delta.data.item(), 5)}, d_w: {round(module.d_delta.data.item(), 5)}')
            if layer_name == 'activation':
                print(f'{name}{layer_name} || c_x: {round(module.c_delta.data.item(), 5)}, d_x: {round(module.d_delta.data.item(), 5)}')

    if EPOCH % 10 == 0:
        check_point = {'state_dict': model.state_dict(),
                       'optim': optimizer.state_dict(),
                       'EPOCH': EPOCH}
        torch.save(check_point, f'./model_{EPOCH+1}.pt')