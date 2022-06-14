import warnings
warnings.filterwarnings('ignore')


import data, models_moco, utils
import argparse, os, time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel


torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser()
parser.add_argument('--port-num',      default='9999', type=str)
parser.add_argument('--world-size',    default=8, type=int, help='number of gpus for ddp')

parser.add_argument('--data-dir',      default='/mnt/ssd1/ImageNet', type=str)
parser.add_argument('--batch-size',    default=256, type=int)
parser.add_argument('--num-workers',   default=8, type=int)

parser.add_argument('--moco-dim',      default=128, type=int, help='feature dimension')
parser.add_argument('--moco-k',        default=65536, type=int, help='queue size; number of negative keys')
parser.add_argument('--moco-m',        default=0.999, type=float, help='moco momentum of updating key encoder')
parser.add_argument('--moco-t',        default=0.2, type=float, help='softmax temperature')

parser.add_argument('--start-epoch',   default=0, type=int)
parser.add_argument('--epochs',        default=200, type=int)
parser.add_argument('--warmup-epochs', default=0, type=int)
parser.add_argument('--min-lr',        default=1e-8, type=float)

parser.add_argument('--lr',            default=0.03, type=float)
parser.add_argument('--momentum',      default=0.9, type=float)
parser.add_argument('--weight-decay',  default=1e-4, type=float)

parser.add_argument('--resume',        default='', type=str, help='latest checkpoint')
parser.add_argument('--save',          action='store_true', help='save logs, checkpoints')
parser.add_argument('--save-name',     default='MoCo_ResNet50_200', type=str)
parser.add_argument('--save-freq',     default=1, type=int)
parser.add_argument('--print-freq',    default=200, type=int)
parser.add_argument('--log',           default='./logs/', type=str)
parser.add_argument('--checkpoint',    default='./checkpoints/', type=str)
args = parser.parse_args()



def init_process(gpu, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = args.port_num
    torch.cuda.set_device(gpu)
    dist.init_process_group('nccl', world_size=world_size, rank=gpu)


def main(gpu, world_size):
    init_process(gpu, world_size)

    # dataloader
    train_dataset = data.PretrainDB(os.path.join(args.data_dir, 'train'))
    train_sampler = DistributedSampler(train_dataset, rank=gpu, num_replicas=args.world_size, shuffle=True, drop_last=True)
    train_loader  = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=int(args.batch_size // args.world_size),
                                                shuffle=False,
                                                sampler=train_sampler,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                drop_last=True)
    
    val_dataset = data.PretrainDB(os.path.join(args.data_dir, 'val'))
    val_sampler = DistributedSampler(val_dataset, rank=gpu, num_replicas=args.world_size, shuffle=True, drop_last=True)
    val_loader  = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=int(args.batch_size // args.world_size),
                                              shuffle=False,
                                              sampler=val_sampler,
                                              num_workers=args.num_workers,
                                              pin_memory=True,
                                              drop_last=True)

    # model
    net = models_moco.MoCo(args.moco_dim, args.moco_k, args.moco_m, args.moco_t).cuda(gpu)
    net = DistributedDataParallel(net, device_ids=[gpu])

    # optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # critertion
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # logger
    logger = utils.Logger(args)
    if dist.get_rank() == 0: logger.initialize()
        
    # resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(gpu))
        args.start_epoch = checkpoint['epoch']
        net.module.encoder_q.load_state_dict(checkpoint['q_state_dict'])
        net.module.encoder_k.load_state_dict(checkpoint['k_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # epoch start
    for epoch in range(args.start_epoch, args.epochs):
        # train
        net.train()
        if dist.get_rank() == 0: print('Epoch {} Train Started...'.format(epoch))

        train_loss = []
        train_start = time.time()
        for i, (imgs_q, imgs_k) in enumerate(train_loader):
            lr = utils.cosine_scheduler(optimizer, epoch + i/len(train_loader), args)

            imgs_q, imgs_k = imgs_q.cuda(gpu), imgs_k.cuda(gpu)
            output, target = net(imgs_q, imgs_k)
            loss = criterion(output, target)

            optimizer.zero_grad(); loss.backward(); optimizer.step()

            dist.barrier()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            if dist.get_rank() == 0: train_loss.append(loss.item() / args.world_size)

            if (i % args.print_freq == 0) and (dist.get_rank() == 0):
                print('Iteration : {:0>5}   LR : {:.6f}   Train Loss : {:.6f}'.format(i, lr, train_loss[-1]))

        train_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start))
        
        
        # Val
        net.eval()
        if dist.get_rank() == 0: print('Epoch {} Val Started...'.format(epoch))
        
        val_start = time.time()
        with torch.no_grad():
            val_loss, val_acc = [], []
            for imgs_q, imgs_k in val_loader:
                imgs_q, imgs_k = imgs_q.cuda(gpu), imgs_k.cuda(gpu)
                output, target = net(imgs_q, imgs_k)
                loss = criterion(output, target)

                N = output.shape[0]
                predict = torch.argmax(output, 1)
                c = (predict == target).sum()
                
                dist.barrier()
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(c, op=dist.ReduceOp.SUM)
                if dist.get_rank() == 0:
                    val_loss.append(loss.item() / args.world_size)
                    val_acc.append(100 * c.item() / (N * args.world_size))
                    
        val_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - val_start))
        

        # Print results
        if dist.get_rank() == 0:
            train_loss = sum(train_loss) / len(train_loss)
            val_loss = sum(val_loss) / len(val_loss)
            acc = sum(val_acc) / len(val_acc)
            print(); print('-' * 50)
            print('Epoch : {}'.format(epoch))
            print('Acc : {:.2f}'.format(acc))
            print('Train Time : {}   Val Time : {}'.format(train_time, val_time))
            print('Train Loss : {:.6f}   Val Loss : {:.6f}'.format(train_loss, val_loss))
            print('-' * 50); print()

            # save checkpoint
            if args.save and (epoch % args.save_freq == 0):
                checkpoint = os.path.join(args.checkpoint, '{}_{:0>4}.pth.tar'.format(args.save_name, epoch))
                torch.save({'epoch' : epoch+1,
                            'q_state_dict' : net.module.encoder_q.state_dict(),
                            'k_state_dict' : net.module.encoder_k.state_dict(),
                            'optimizer' : optimizer.state_dict()}, 
                             checkpoint)

            # update log
            logger.update({'epoch' : epoch,
                           'lr' : lr,
                           'acc' : acc,
                           'train_time' : train_time,
                           'train_loss' : train_loss,
                           'val_time' : val_time,
                           'val_loss' : val_loss,})



def run(world_size):
    torch.multiprocessing.spawn(main, nprocs=world_size, args=(world_size,))
    dist.destroy_process_group()


if __name__ == '__main__':
    print('Available GPUs : {}   Use GPUs : {}'.format(torch.cuda.device_count(), args.world_size))
    assert args.world_size <= torch.cuda.device_count()
    run(args.world_size)