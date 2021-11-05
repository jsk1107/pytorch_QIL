import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import transforms as T
from resnet import resnet20, resnet18
import torch.optim as optim
from tqdm import tqdm
from utils import Hook, interval_param_list, weight_param_list, check_interval_param, graph_prun_ratio
from qil import Transformer
from logger import get_logger
import argparse



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


def train(model, criterion, optimizer, lr_scheduler, train_loader, device, EPOCH):

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
                f'EPOCH: {EPOCH + 1} | total_train_loss: {train_loss / (i + 1):.4f} | batch_loss: {iter_loss:.4f}'
            )
        lr_scheduler.step(EPOCH)


def validation(model, criterion, device, val_loader, EPOCH):

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

    logger.info(f'EPOCH: {EPOCH + 1} | '
                f'Loss: {val_loss / (i + 1):.4f} | '
                f'Accuracy: {acc:.4f}%')

    return acc


def main(args, **kwargs):

    """ CONFIG """
    w_bit, a_bit = kwargs.get('w_bit'), kwargs.get('a_bit')
    pretrain_model_path = args.pretrained_path
    batch_size = args.batch_size

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        raise NotImplementedError('GPU 셋팅을 다시 해주세요.')
    """ Data Type 셋팅 """
    if args.data == 'ILSVRC2012':
        if w_bit == 32 and a_bit == 32:
            EPOCHS = 120
            interval_lr = 0
            weight_lr = 4e-1
            milestones = [30, 60, 85, 95, 105]
        else:
            EPOCHS = 90
            interval_lr = 4e-4
            weight_lr = 4e-2
            milestones = [20, 40, 60, 80]
        train_set = ImageFolder('../data/imagenet/', transform=custom_transfroms_imagenet(True))
        val_set = ImageFolder('../data/imagenet/', transform=custom_transfroms_imagenet(False))
        model = resnet18(pretrain_model_path, **kwargs)

    elif args.data == 'CIFAR10':
        if w_bit == 32 and a_bit == 32:
            EPOCHS = 200
            interval_lr = 0
            weight_lr = 4e-1
            milestones = [100, 150, 180]
        else:
            EPOCHS = 150
            interval_lr = 4e-4
            weight_lr = 4e-2
            milestones = [60, 90, 120]
        train_set = CIFAR10('../data', train=True, transform=custom_transfroms_cifar10(True), download=True)
        val_set = CIFAR10('../data', train=False, transform=custom_transfroms_cifar10(False))
        model = resnet20(pretrain_model_path, **kwargs)
    else:
        raise NotImplementedError()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    """ 모델 셋팅 """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': interval_param_list(model), 'lr': interval_lr},
                           {'params': weight_param_list(model), 'lr': weight_lr}], momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    model.to(device)


    if w_bit != 32 and a_bit != 32:

        """ Hook 등록 """
        handle_hooks = []
        for module_name, module in model.named_modules():
            if isinstance(module, Transformer):
                h = Hook(module, module_name)
                handle_hooks.append(h)

        """ Batch_size만큼 1회 Iteration후 c_x, d_x 초기값 정하기 """
        model.train()
        tmp_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
        data = next(iter(tmp_dataloader))
        optimizer.zero_grad()
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        output = model(imgs)
        iter_loss = criterion(output, targets)
        iter_loss.backward()
        print('c_delta, d_delta 초기값 셋팅 완료')

        """ Check Interval Param """
        check_interval_param(model)

        """ Prun_ratio 그래프 그리기 """
        graph_prun_ratio(handle_hooks, plot=True)

        """ 등록했던 Hook 제거 """
        for handle_hook in handle_hooks:
            handle_hook.handle.remove()

    best_acc = .0
    for EPOCH in range(EPOCHS):
        train(model, criterion, optimizer, lr_scheduler, train_loader, device, EPOCH)

        if EPOCH % 5 == 0:
            acc = validation(model, criterion, device, val_loader, EPOCH)

            """ Interval Param Check """
            check_interval_param(model)

            if best_acc < acc:
                best_acc = acc
                check_point = {'state_dict': model.state_dict(),
                               'optim': optimizer.state_dict(),
                               'EPOCH': EPOCH,
                               'Acc': best_acc.round()}
                torch.save(check_point, f'./final_{kwargs.get("w_bit")}_{kwargs.get("a_bit")}_model.pt')
    logger.info(f'Best_Acc: {best_acc:.4f}%')


if __name__ == '__main__':

    parser = argparse.ArgumentParser("QIL")
    parser.add_argument('-d', '--data', type=str, default='CIFAR10', help='데이터셋 종류. ILSVRC2012 or CIFAR10')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--use-cuda', type=bool, default=True, help='쿠다 사용')
    parser.add_argument('--pretrained-path', type=str, default='./final_32_32_model.pt', help='Pretrained Model의 경로 지정')
    args = parser.parse_args()

    bits = [[32, 32], [5, 5], [4, 4], [3, 3], [2, 2]]
    for i in bits:
        kwargs = {'w_bit': i[0], 'a_bit': i[1]}
        main(args, **kwargs)