import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import transforms as T
from resnet import resnet20, resnet18
import torch.optim as optim
from tqdm import tqdm
from utils import Hook, interval_param_list, weight_param_list, check_interval_param, prun_ratio
from qil import Transformer
from logger import get_logger
import argparse
import os
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel, DataParallel
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist


""" Logger 등록 """
logging = get_logger('./log', log_config='./logging.json')
model_logger = logging.getLogger('model-logger')
param_logger = logging.getLogger('param-logger')

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
        return T.Compose([ T.RandomResizedCrop(224),
                           T.RandomHorizontalFlip(),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                           ])
    else:
        return T.Compose([ T.Resize(256),
                           T.CenterCrop(224),
                           T.ToTensor(),
                           T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                           ])


def train(model, criterion, optimizer, lr_scheduler, train_loader, EPOCH, args):

    model.train()
    train_loss = .0
    with tqdm(train_loader) as tbar:
        for i, data in enumerate(tbar):
            optimizer.zero_grad()

            if args.gpu is not None:
                imgs = imgs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            else:
                imgs, targets = imgs.to(args.device), targets.to(args.device)

            output = model(imgs)
            iter_loss = criterion(output, targets)
            train_loss += iter_loss

            iter_loss.backward()
            optimizer.step()

            tbar.set_description(
                f'EPOCH: {EPOCH} | total_train_loss: {train_loss / (i + 1):.4f} | batch_loss: {iter_loss:.4f}'
            )
        lr_scheduler.step()


def validation(model, criterion, val_loader, EPOCH, args):

    model.eval()
    val_loss = .0
    correct = .0
    total = 0

    with tqdm(val_loader) as tbar:
        for i, data in enumerate(tbar):
            imgs, targets = data

            if args.gpu is not None:
                imgs = imgs.cuda(args.gpu, non_blocking=True)
                targets = targets.cuda(args.gpu, non_blocking=True)
            else:
                imgs, targets = imgs.to(args.device), targets.to(args.device)

            with torch.no_grad():
                output = model(imgs)
            iter_loss = criterion(output, targets)
            val_loss += iter_loss

            tbar.set_description(
                f'EPOCH: {EPOCH} | total_val_loss: {val_loss / (i + 1):.4f} | batch_loss: {iter_loss:.4f}')

            _, pred_idx = torch.max(output, dim=1)
            total += targets.size(0)
            correct += (pred_idx == targets).sum()
    acc = 100 * correct / total

    model_logger.logger.info(f'EPOCH: {EPOCH} | '
                             f'Loss: {val_loss / (i + 1):.4f} | '
                             f'Accuracy: {acc:.4f}%')

    return acc


def initialize_param(model, train_set, train_sampler, interval_lr, weight_lr):

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD([{'params': interval_param_list(model), 'lr': interval_lr},
                           {'params': weight_param_list(model), 'lr': weight_lr}], momentum=0.9, weight_decay=1e-4)
    """ Hook """
    handle_hooks = []
    for module_name, module in model.named_modules():
        if isinstance(module, Transformer):
            h = Hook(module, module_name, flag=True)
            handle_hooks.append(h)

    """ Batch_size만큼 Iteration 1회를 통해서 c_x, d_x 초기값 구하기 """

    model.to(args.device)
    model.train()
    tmp_dataloader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=(train_sampler is None),
                                sampler=train_sampler, pin_memory=True)

    """ c_delta, d_delta 초기값 설정 """
    data = next(iter(tmp_dataloader))
    optimizer.zero_grad()
    imgs, targets = data
    if args.multiprocessing_distributed:
        if args.gpu is not None:
            imgs = imgs.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            targets = targets.cuda(args.gpu, non_blocking=True)
    else:
        imgs = imgs.to(args.device)
        targets = targets.to(args.device)

    output = model(imgs)
    iter_loss = criterion(output, targets)
    iter_loss.backward()


    """ 등록된 hook 제거(메모리관리) """
    for handle_hook in handle_hooks:
        handle_hook.handle.remove()

    print('c_delta, d_delta 초기값 설정 완료'    )

    """ Check Interval Param """
    check_interval_param(param_logger, model)

    """ Prun_ratio ▒~D▒~B▒▒~U~X기 """
    prun_ratio(handle_hooks, plot=True)

    criterion = None
    optimizer = None
    lr_scheduler = None

def main(args, **kwargs):

    num_device_per_node = torch.cuda.device_count()
    args.multiprocessing_distributed = False
    if args.multiprocessing_distributed:
        args.world_size = num_device_per_node * args.world_size
        mp.spawn(main_worker, nprocs=num_device_per_node, args=(num_device_per_node, args, kwargs))
    else:
        main_worker(args.gpu, num_device_per_node, args, kwargs)


def main_worker(gpu, num_device_per_node, args, **kwargs):

    """ CONFIG """
    w_bit, a_bit = kwargs.get('w_bit'), kwargs.get('a_bit')

    if w_bit == 32:
        pass
    elif w_bit == 5:
        args.pretrain_model_path = f'best_32_32_model.pt'
    elif w_bit == 4:
        args.pretrain_model_path = 'best_5_5_model.pt'
    elif w_bit == 3:
        args.pretrain_model_path = 'best_4_4_model.pt'
    elif w_bit == 2:
        args.pretrain_model_path = 'best_3_3_model.pt'

    if os.path.exists(args.pretrain_model_path):
        print(f'pretrain_model_path: {args.pretrain_model_path}')

    args.gpu = gpu
    kwargs['gpu'] = gpu


    if args.gpu is not None:
        print(f'Use GPU: {args.gpu} for training')

    """ 멀티프로세싱 분산처리 """
    if args.multiprocessing_distributed:
        args.rank = args.rank * num_device_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    else:
        args.deviceIds = [int(i) for i in range(num_device_per_node)]
        args.device = torch.device('cuda')

    """ Data Type 셋팅 """
    if args.data == 'ILSVRC2012':
        """ The same environment as the setting in the paper """
        batch_size = args.batch_size
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
        train_set = ImageFolder(args.image_dir, transform=custom_transfroms_imagenet(True))
        val_set = ImageFolder(args.image_dir, transform=custom_transfroms_imagenet(False))
        model = resnet18(args.pretrain_model_path, num_classes=1000, **kwargs)

    elif args.data == 'CIFAR10':
        """ 내가 설정한 임의의 환경 """
        batch_size = 128
        if w_bit == 32 and a_bit == 32:
            EPOCHS = 200
            interval_lr = 0
            weight_lr = 1e-1
            milestones = [100, 150, 180]
        else:
            EPOCHS = 150
            interval_lr = 1e-4
            weight_lr = 1e-2
            milestones = [60, 90, 120]
        train_set = CIFAR10('../data', train=True, transform=custom_transfroms_cifar10(True), download=True)
        val_set = CIFAR10('../data', train=False, transform=custom_transfroms_cifar10(False))
        model = resnet20(args.pretrain_model_path, num_classes=10, **kwargs)
    else:
        raise NotImplementedError()

    if args.multiprocessing_distributed:
        train_sampler = DistributedSampler(train_set)
    else:
        train_sampler = None



    if args.gpu is None:
        if w_bit != 32 and a_bit != 32:
            initialize_param(model, train_set, train_sampler, interval_lr, weight_lr)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': interval_param_list(model), 'lr': interval_lr},
                           {'params': weight_param_list(model), 'lr': weight_lr}], momentum=0.9, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    """ 멀티프로세싱 분산처리 여부에 따른 모델 설졍 """
    if args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / num_device_per_node)
            args.workers = int((args.workers + num_device_per_node - 1) / num_device_per_node)
            model = DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = DataParallel(model, device_ids=args.deviceIds).to(device=args.device)

    print('model type is DataParallel')

    print('workers', args.workers)
    print('batch_size', args.batch_size)


    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.workers,
                              shuffle=(train_sampler is None),
                              sampler=train_sampler, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers, shuffle=False,
                            pin_memory=True)

    cudnn.benchmark = True

    best_acc = .0
    model_logger.info(f'{args.exp}_Start: {w_bit} | {a_bit}')
    param_logger.info(f'{args.exp}_Start: {w_bit} | {a_bit}')
    check_interval_param(param_logger, model)

    for EPOCH in range(1, EPOCHS + 1):
        """ EPOCH마다 GPU 할당되는 인덱스 리스트 설정 """
        if args.multiprocessing_distributed:
            train_sampler.set_epoch(EPOCH)
        train(model, criterion, optimizer, lr_scheduler, train_loader, EPOCH, args)

        if EPOCH == 1 or EPOCH % 5 == 0:
            acc = validation(model, criterion, val_loader, EPOCH, args)

            """ Check Interval Param """
            if w_bit != 32 and a_bit != 32:
                param_logger.info(f'{args.exp}_EPOCH: {EPOCH}')
                check_interval_param(param_logger, model)

            """ Save Model """
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % num_device_per_node == 0):
                if best_acc < acc:
                    best_acc = acc
                    check_point = {'state_dict': model.module.state_dict(),
                               'optim': optimizer.state_dict(),
                               'EPOCH': EPOCH,
                               'Acc': best_acc.round()}
                    torch.save(check_point, f'./best_{kwargs.get("w_bit")}_{kwargs.get("a_bit")}_model.pt')
    model_logger.info(f'Best_Acc: {best_acc:.4f}%')


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Ex")
    parser.add_argument('-d', '--data', type=str, default='ILSVRC2012', help='ILSVRC2012 or CIFAR10 or CIFAR100')
    parser.add_argument('-b', '--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--multiprocessing-distributed', action='store_true', help='병렬처리를 위해 Multi-GPU 사용')
    parser.add_argument('--gpu', default=None, type=int, help='사용할 GPU Id')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:54321', type=str, help='분산처리를 사용하기 위해 사용되는 URL')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int, help='분산처리를 하기 위한 노드의 수')
    parser.add_argument('--rank', default=0, type=int, help='분산학습을 위한 노드의 랭크')
    parser.add_argument('--pretrained-path', type=str, default='./best_32_32_model.pt', help='Pretrained Model의 경로 지정')
    parser.add_argument('--image-dir', '-i', type=str, default='../imagenet/', help='imagenet data dir')
    parser.add_argument('--exp', type=str, default='qil', help='quantization method')
    args = parser.parse_args()
    bits = [[5, 5], [4, 4], [3, 3], [2, 2]]
    model_logger.info(f'Experiment: {args.exp}')
    for i in bits:
        kwargs = {'w_bit': i[0], 'a_bit': i[1]}
        main(args, **kwargs)
