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

kwarg = {'w_bit': 32, 'a_bit': 32}

model = resnet20(**kwarg)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 180], gamma=0.1)
model.to(device)
#
# model_path = './model_6.pt'
# if model_path:
#     print(f'load: {kwarg} | {model_path}')
#
#     checkpoint = torch.load(model_path)
#     model.load_state_dict(checkpoint['state_dict'], strict=True)
#     optimizer.load_state_dict(checkpoint['optim'])

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

    # if EPOCH % 10 == 0:
    check_point = {'state_dict': model.state_dict(),
                   'optim': optimizer.state_dict()}
    torch.save(check_point, f'./model_{EPOCH+1}.pt')