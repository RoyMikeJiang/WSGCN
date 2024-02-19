from __future__ import print_function, absolute_import, division

import os
import time
import datetime
import argparse
import numpy as np
import os.path as path
import itertools

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from progress.bar import Bar
from common.camera import project_to_2d, project_to_2d_linear
from common.log import Logger, savefig
from common.utils import AverageMeter, lr_decay, save_ckpt
from common.graph_utils import adj_mx_from_skeleton
from common.data_utils import fetch, read_3d_data, create_2d_data
from common.generators import PoseGenerator, UnlabeledPoseGenerator
from common.loss import mpjpe, p_mpjpe
from models.sem_gcn import SemGCN


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')

    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset')
    parser.add_argument('-k', '--keypoints', default='gt', type=str, metavar='NAME', help='2D detections to use')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME',
                        help='checkpoint to evaluate (file name)')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME',
                        help='checkpoint to resume (file name)')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--snapshot', default=5, type=int, help='save models for every #snapshot epochs (default: 20)')

    # Model arguments
    parser.add_argument('-l', '--num_layers', default=4, type=int, metavar='N', help='num of residual layers')
    parser.add_argument('-z', '--hid_dim', default=128, type=int, metavar='N', help='num of hidden dimensions')
    parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N',
                        help='batch size in terms of predicted frames')
    parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N', help='number of training epochs')
    parser.add_argument('--num_workers', default=4, type=int, metavar='N', help='num of workers for data loading')
    parser.add_argument('--lr', default=1.0e-3, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', type=int, default=100000, help='num of steps of learning rate decay')
    parser.add_argument('--lr_gamma', type=float, default=0.96, help='gamma of learning rate decay')
    parser.add_argument('--no_max', dest='max_norm', action='store_false', help='if use max_norm clip on grad')
    parser.set_defaults(max_norm=True)
    parser.add_argument('--non_local', dest='non_local', action='store_true', help='if use non-local layers')
    parser.set_defaults(non_local=False)
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate')

    # Experimental
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor')

    # Semi_supervision
    parser.add_argument('--warm_up', default=20, type=int, metavar='N', help='warm up epoch')

    # Semi_bonelosdd
    parser.add_argument('--bone_loss', default=True, type=bool, help='bone loss activator')

    args = parser.parse_args()

    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()

    return args


def main(args):
    print('==> Using settings {}'.format(args))

    print('==> Loading dataset...')
    dataset_path = path.join('data', 'data_3d_' + args.dataset + '.npz')
    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset, TRAIN_SUBJECTS, TEST_SUBJECTS, FULL_SUBJECTS, SEMI_SUBJECTS
        dataset = Human36mDataset(dataset_path)
        subjects_train = TRAIN_SUBJECTS
        subjects_test = TEST_SUBJECTS
        subjects_full_train = FULL_SUBJECTS
        subjects_semi_train = SEMI_SUBJECTS
        
    else:
        raise KeyError('Invalid dataset')

    print('==> Preparing data...')
    dataset = read_3d_data(dataset)

    print('==> Loading 2D detections...')
    keypoints = create_2d_data(path.join('data', 'data_2d_' + args.dataset + '_' + args.keypoints + '.npz'), dataset)

    action_filter = None if args.actions == '*' else args.actions.split(',')
    if action_filter is not None:
        action_filter = map(lambda x: dataset.define_actions(x)[0], action_filter)
        print('==> Selected actions: {}'.format(action_filter))

    stride = args.downsample
    cudnn.benchmark = True
    device = torch.device("cuda")

    # Create model
    print("==> Creating model...")

    p_dropout = (None if args.dropout == 0.0 else args.dropout)
    adj = adj_mx_from_skeleton(dataset.skeleton())
    model_pos = SemGCN(adj, args.hid_dim, num_layers=args.num_layers, p_dropout=p_dropout,
                       nodes_group=dataset.skeleton().joints_group() if args.non_local else None).to(device)
    print("==> Total parameters: {:.2f}M".format(sum(p.numel() for p in model_pos.parameters()) / 1000000.0))

    criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = torch.optim.Adam(model_pos.parameters(), lr=args.lr)

    # Optionally resume from a checkpoint
    if args.resume or args.evaluate:
        ckpt_path = (args.resume if args.resume else args.evaluate)

        if path.isfile(ckpt_path):
            print("==> Loading checkpoint '{}'".format(ckpt_path))
            ckpt = torch.load(ckpt_path)
            start_epoch = ckpt['epoch']
            error_best = ckpt['error']
            glob_step = ckpt['step']
            lr_now = ckpt['lr']
            # for k, v in ckpt['state_dict'].copy().items():
            #     ckpt['state_dict'][k.replace('nonlocal', 'non_local')] = ckpt['state_dict'].pop(k)

            model_pos.load_state_dict(ckpt['state_dict'], strict=False)
            # model_pos.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            print("==> Loaded checkpoint (Epoch: {} | Error: {})".format(start_epoch, error_best))

            if args.resume:
                ckpt_dir_path = path.dirname(ckpt_path)
                logger = Logger(path.join(ckpt_dir_path, 'log.txt'), resume=True)
        else:
            raise RuntimeError("==> No checkpoint found at '{}'".format(ckpt_path))
    else:
        start_epoch = 0
        error_best = None
        glob_step = 0
        lr_now = args.lr
        ckpt_dir_path = path.join(args.checkpoint, datetime.datetime.now().isoformat())

        if not path.exists(ckpt_dir_path):
            os.makedirs(ckpt_dir_path)
            print('==> Making checkpoint dir: {}'.format(ckpt_dir_path))

        logger = Logger(os.path.join(ckpt_dir_path, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'error_eval_p1', 'error_eval_p2'])

    if args.evaluate:
        print('==> Evaluating...')

        if action_filter is None:
            action_filter = dataset.define_actions()

        errors_p1 = np.zeros(len(action_filter))
        errors_p2 = np.zeros(len(action_filter))

        for i, action in enumerate(action_filter):
            poses_valid, poses_valid_2d, actions_valid = fetch(subjects_test, dataset, keypoints, [action], stride)
            valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid),
                                      batch_size=args.batch_size, shuffle=False,
                                      num_workers=args.num_workers, pin_memory=True)
            errors_p1[i], errors_p2[i] = evaluate(valid_loader, model_pos, device)

        print('Protocol #1   (MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p1).item()))
        print('Protocol #2 (P-MPJPE) action-wise average: {:.2f} (mm)'.format(np.mean(errors_p2).item()))
        exit(0)

    # full train dataloader
    poses_full_train, poses_full_train_2d, actions_full_train, _= fetch(subjects_full_train, dataset, keypoints, action_filter, stride)
    full_train_loader = DataLoader(PoseGenerator(poses_full_train, poses_full_train_2d, actions_full_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # semi train dataloader
    poses_semi_train, poses_semi_train_2d, actions_semi_train, semi_out_cam_params = fetch(subjects_semi_train, dataset, keypoints, action_filter, stride)
    semi_train_loader = DataLoader(UnlabeledPoseGenerator(poses_semi_train_2d,semi_out_cam_params, actions_semi_train, poses_semi_train), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    poses_valid, poses_valid_2d, actions_valid, _ = fetch(subjects_test, dataset, keypoints, action_filter, stride)
    valid_loader = DataLoader(PoseGenerator(poses_valid, poses_valid_2d, actions_valid), batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(len(full_train_loader))
    print(len(semi_train_loader))
    print(len(valid_loader))

    for epoch in range(start_epoch, args.epochs):
        if epoch < args.warm_up:
            warm_up = True
        else:
            warm_up = False

        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr_now))

        # Train for one epoch
        # epoch_loss, lr_now, glob_step = train(train_loader, model_pos, criterion, optimizer, device, args.lr, lr_now,
        #                                       glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm)
        epoch_loss, lr_now, glob_step = train(full_train_loader, semi_train_loader, model_pos, criterion, optimizer, device, args.lr, lr_now,
                                              glob_step, args.lr_decay, args.lr_gamma, max_norm=args.max_norm, warm_up=warm_up, skeleton_parents=dataset.skeleton().parents() if args.bone_loss==True else None)

        # Evaluate
        error_eval_p1, error_eval_p2 = evaluate(valid_loader, model_pos, device)

        # Update log file
        logger.append([epoch + 1, lr_now, epoch_loss, error_eval_p1, error_eval_p2])

        # Save checkpoint
        if error_best is None or error_best > error_eval_p1:
            error_best = error_eval_p1
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path, suffix='best')

        if (epoch + 1) % args.snapshot == 0:
            save_ckpt({'epoch': epoch + 1, 'lr': lr_now, 'step': glob_step, 'state_dict': model_pos.state_dict(),
                       'optimizer': optimizer.state_dict(), 'error': error_eval_p1}, ckpt_dir_path)

    logger.close()
    logger.plot(['loss_train', 'error_eval_p1'])
    savefig(path.join(ckpt_dir_path, 'log.eps'))

    return

class InfiniteRandomIterator:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(loader)
        self.batch_queue = []

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.batch_queue) == 0:
            try:
                batch = next(self.iterator)
                self.batch_queue.append(batch)
            except StopIteration:
                self.iterator = iter(self.loader)
                # random.shuffle(self.loader.dataset)
                batch = next(self.iterator)
                self.batch_queue.append(batch)

        return self.batch_queue.pop(0)

# def train(data_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True):
def train(data_loader, semi_train_loader, model_pos, criterion, optimizer, device, lr_init, lr_now, step, decay, gamma, max_norm=True, warm_up=False, skeleton_parents=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_total_loss = AverageMeter()

    # Switch to train mode
    torch.set_grad_enabled(True)
    model_pos.train()
    end = time.time()

    bar = Bar('Train', max=len(semi_train_loader))

    data_loader_iter = InfiniteRandomIterator(data_loader)

    for i, (semi_inputs_2d, semi_3d_gt, cams) in enumerate(semi_train_loader):

        # Measure data loading time
        data_time.update(time.time() - end)

        targets_3d, inputs_2d, _ = next(data_loader_iter)
        num_poses = targets_3d.size(0)

        step += 1
        if step % decay == 0 or step == 1:
            lr_now = lr_decay(optimizer, step, lr_init, decay, gamma)

        targets_3d, inputs_2d = targets_3d.to(device), inputs_2d.to(device)
        
        inputs_traj = targets_3d[:, :1].clone()
        targets_3d[:, 0] = 0

        outputs_3d = model_pos(inputs_2d)

        optimizer.zero_grad()
        loss_3d_pos = criterion(outputs_3d, targets_3d)
        # loss_3d_pos = mpjpe(outputs_3d, targets_3d)
        # epoch_loss_3d = loss_3d_pos*num_poses
        loss_total = loss_3d_pos
        if not warm_up:
            cams = cams.to(device)
            semi_inputs_2d = semi_inputs_2d.to(device)
            semi_3d_gt = semi_3d_gt.to(device)
            semi_num_poses = semi_inputs_2d.size(0)

            semi_output_3d = model_pos(semi_inputs_2d)

            semi_reconstruction_2d = project_to_2d(semi_output_3d+semi_3d_gt[:,0].unsqueeze(1), cams)
            semi_loss_2d = criterion(semi_reconstruction_2d, semi_inputs_2d)
            # semi_loss_2d = mpjpe(semi_reconstruction_2d, semi_inputs_2d)
            # epoch_loss_semi_2d = semi_loss_2d*semi_num_poses

            loss_total += semi_loss_2d

            # Bone length term to enforce kinematic constraints
            if True:
                dists_full = outputs_3d[:, 1:, :] - outputs_3d[:, skeleton_parents[1:], :]
                dists_semi = semi_output_3d[:, 1:, :] - semi_output_3d[:, skeleton_parents[1:], :]
                bone_lengths_full = torch.norm(dists_full, dim=2)
                bone_lengths_semi = torch.norm(dists_semi, dim=2)
                penalty = torch.mean(torch.abs(torch.mean(bone_lengths_full, dim=0) \
                                             - torch.mean(bone_lengths_semi, dim=0)))
                loss_total += penalty
        loss_total.backward()
        if max_norm:
            nn.utils.clip_grad_norm_(model_pos.parameters(), max_norm=1)
        optimizer.step()

        epoch_total_loss.update(loss_total.item(), 1)

        # Measure elapsed times
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| Loss: {loss: .4f}' \
            .format(batch=i + 1, size=len(semi_train_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, loss=epoch_total_loss.avg)
        bar.next()

    bar.finish()
    return epoch_total_loss.avg, lr_now, step


def evaluate(data_loader, model_pos, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    epoch_loss_3d_pos = AverageMeter()
    epoch_loss_3d_pos_procrustes = AverageMeter()

    # Switch to evaluate mode
    torch.set_grad_enabled(False)
    model_pos.eval()
    end = time.time()

    bar = Bar('Eval ', max=len(data_loader))
    for i, (targets_3d, inputs_2d, _) in enumerate(data_loader):
        # Measure data loading time
        data_time.update(time.time() - end)
        num_poses = targets_3d.size(0)

        inputs_2d = inputs_2d.to(device)
        outputs_3d = model_pos(inputs_2d).cpu()
        inputs_traj = targets_3d[:, :1].clone()
        targets_3d[:, 0] = 0
        outputs_3d[:, :, :] -= outputs_3d[:, :1, :]  # Zero-centre the root (hip)

        epoch_loss_3d_pos.update(mpjpe(outputs_3d, targets_3d).item() * 1000.0, num_poses)
        epoch_loss_3d_pos_procrustes.update(p_mpjpe(outputs_3d.numpy(), targets_3d.numpy()).item() * 1000.0, num_poses)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {ttl:} | ETA: {eta:} ' \
                     '| MPJPE: {e1: .4f} | P-MPJPE: {e2: .4f}' \
            .format(batch=i + 1, size=len(data_loader), data=data_time.avg, bt=batch_time.avg,
                    ttl=bar.elapsed_td, eta=bar.eta_td, e1=epoch_loss_3d_pos.avg, e2=epoch_loss_3d_pos_procrustes.avg)
        bar.next()

    bar.finish()
    return epoch_loss_3d_pos.avg, epoch_loss_3d_pos_procrustes.avg


if __name__ == '__main__':
    main(parse_args())
