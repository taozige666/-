import time
import argparse
import datetime

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import numpy as np
from torchvision import transforms
from pytorch_msssim import MS_SSIM

from loss import SILogLoss, BinsChamferLoss, GradientLoss
from model.ViT_adaptive_bins import ViTAdaBins
from dataloader import getTrainingTestingData, getTestData
from utils import AverageMeter, DepthNorm, colorize


def weights_init(m):
    # if hasattr(m, 'weight'):
    #     nn.init.xavier_normal_(m.weight.data) 
    #     print("inited")
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        print("Conv2d Layer Inited")
    elif classname.find('BatchNorm2d') != -1:
        nn.init.constant_(m.weight.data, 0.01)
        nn.init.constant_(m.bias.data, 0)
        print("BatchNorm2d Layer Inited")
    elif classname.find('LayerNorm') != -1:
        nn.init.constant_(m.weight.data, 0.01)
        nn.init.constant_(m.bias.data, 0)
        print("LayerNorm Layer Inited")
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        print("Linear Layer Inited")


def main():
    # Arguments
    parser = argparse.ArgumentParser(description='ViT and Adaptive Bins for Monocular Depth Estimation')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=2, type=int, help='batch size')
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument('--n_bins', default=256, type=int, help='number of bins to divide depth range')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")

    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', default=False, help='if set, crops according to Garg  ECCV16',
                        action='store_true')
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    args = parser.parse_args()
    device = torch.device('cuda')

    # create model
    model = ViTAdaBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm).to(
        device)
    print('Model created.')
    print(model)
    optimizer = torch.optim.Adam([
        {'params': model.pretrained.model.parameters(), 'lr': 1e-5},
        {'params': model.scratch.parameters(), 'lr': 1e-4},
        {'params': model.adaptive_bins_layer.parameters(), 'lr': 2e-4},
        {'params': model.conv_out.parameters(), 'lr': 1e-4}
    ], lr=args.lr)
    # 阶梯衰减，每次衰减的epoch数根据列表 [20, 30, 60, 80] 给出，0.8代表学习率衰减倍数
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [4, 10, 16, 19], 0.8)
    model.apply(weights_init)
    has_checkpoint = False
    if has_checkpoint:
        checkpoint = torch.load('./MyDPT_gradient.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        model.scratch.apply(weights_init)
        model.adaptive_bins_layer.apply(weights_init)
        model.conv_out.apply(weights_init)
        print("#####weights inited#####")
        start_epoch = -1

    has_pretrained = True
    if has_pretrained:
        # import timm
        # pretrained = timm.create_model('vit_large_patch16_384', pretrained=True)

        pretrained = torch.load('pretrained/jx_vit_base_resnet50_384-9fd3c705.pth')
        model.pretrained.model.load_state_dict(pretrained, strict=True)
        print("#####pretrain loaded#####")

    # Logging
    prefix = 'VisionTransformer_' + str(args.bs)
    writer = SummaryWriter(comment='{}-lr{}-e{}-bs{}'.format(prefix, args.lr, args.epochs, args.bs), flush_secs=30)

    # Training parameters
    batch_size = args.bs

    # loss
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss()
    criterion_gradient = GradientLoss()

    # load data
    train_loader, test_loader = getTrainingTestingData(batch_size=batch_size)
    best_rms = 999
    # start training...
    for epoch in range(start_epoch + 1, args.epochs):
        batch_time = AverageMeter()
        losses = AverageMeter()
        N = len(train_loader)
        end = time.time()

        for i, sample_batched in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # Normalize depth
            depth_n = DepthNorm(depth)

            # Predict
            bin_edges, output = model(image)

            # Compute the loss
            mask = depth_n > args.min_depth
            l_dense = criterion_ueff(output, depth_n, mask=mask.to(torch.bool), interpolate=True)
            l_chamfer = criterion_bins(bin_edges, depth_n)
            # l_gradient = criterion_gradient(output, depth_n)
            ms_ssim_module = MS_SSIM(data_range=10, size_average=True, channel=1)
            ssim_loss = 1 - ms_ssim_module(output, depth_n)
            ssim_loss = 10 * torch.sqrt(ssim_loss)
            # print(l_dense.data.item(),l_chamfer.data.item(), ssim_loss.data.item(), l_gradient.data.item())
            loss = l_dense + ssim_loss + args.w_chamfer * l_chamfer

            # Update step
            losses.update(loss.data.item(), image.size(0))
            loss.backward()
            optimizer.step()
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            eta = str(datetime.timedelta(seconds=int(batch_time.val * (N - i))))

            # Log progress
            niter = epoch * N + i
            if i % 5 == 0:
                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t'
                      'ETA {eta}\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'
                      .format(epoch, i, N, batch_time=batch_time, loss=losses, eta=eta))
                # print(optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], optimizer.param_groups[2]['lr'])

                # Log to tensorboard
                writer.add_scalar('Train/Loss', losses.val, niter)
            # if i == 100:
            #   break
            if i % 300 == 0:
                LogProgress(model, writer, test_loader, niter)

        scheduler.step()
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                 'scheduler': scheduler.state_dict()}
        torch.save(state, './MyDPT_gradient.pth')
        rmse = evaluate(model, device, writer, epoch, args)
        if rmse < best_rms:
            best_rms = rmse
            best_rmse_state_dict = model.state_dict()
            torch.save(best_rmse_state_dict, './epoch{0}_best_rmse_params.pt'.format(epoch))
        # Record epoch's intermediate results
        LogProgress(model, writer, test_loader, niter)
        writer.add_scalar('Train/Loss.avg', losses.avg, epoch)

    torch.save(model.state_dict(), './MyDPT_params.pt')


def evaluate(model, device, writer, epoch, args):
    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))

        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()

        return a1, a2, a3, abs_rel, rmse, log_10

    test_loader = getTestData(batch_size=1)
    depth_scores = np.zeros((6, len(test_loader)))  # six metrics
    with torch.no_grad():
        for i, sample_batched in enumerate(test_loader):
            model.eval()
            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            depth_n = depth.cpu().numpy() / 1000
            pred = model(image)[-1].cpu().numpy()
            pred = DepthNorm(pred) / 100
            flip_trans = transforms.RandomHorizontalFlip(p=1)
            image_flip = flip_trans(image)
            pred_flip = model(image_flip)[-1].cpu().numpy()
            pred_flip = DepthNorm(pred_flip) / 100
            valid_mask = np.logical_and(depth_n > 1e-3, depth_n < 10)

            if args.garg_crop or args.eigen_crop:
                _, _, gt_height, gt_width = depth_n.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[:, :, int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1


                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[:, :, int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[:, :, 45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)

            final = (0.5 * pred) + (0.5 * flip_trans(torch.Tensor(pred_flip)).numpy())
            final = torch.nn.functional.interpolate(torch.Tensor(final), image.shape[-2:], mode='bilinear',
                                                    align_corners=True).numpy()
            errors = compute_errors(depth_n[valid_mask], final[valid_mask])
            print(i)
            print(errors[4])
            for k in range(len(errors)):
                depth_scores[k][i] = errors[k]

        e = depth_scores.mean(axis=1)
        writer.add_scalar('epoch/a1', e[0], epoch)
        writer.add_scalar('epoch/a2', e[1], epoch)
        writer.add_scalar('epoch/a3', e[2], epoch)
        writer.add_scalar('epoch/rel', e[3], epoch)
        writer.add_scalar('epoch/rms', e[4], epoch)
        writer.add_scalar('epoch/log_10', e[5], epoch)
        return e[4]


def LogProgress(model, writer, test_loader, epoch):
    model.eval()
    sequential = test_loader
    sample_batched = next(iter(sequential))
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True)) / 1000
    depth_n = DepthNorm(depth)
    # print(depth, "\n\n\n",depth_n)
    if epoch == 0:
        writer.add_image('Train.1.Image', vutils.make_grid(image.data, nrow=6, normalize=True), epoch)
    if epoch == 0:
        writer.add_image('Train.2.Depth', colorize(vutils.make_grid(depth_n.data, nrow=6, normalize=False)), epoch)
    _, output = model(image)
    output = DepthNorm(output) / 100
    output = DepthNorm(output)
    # print(output, "\n\n\n",depth_n[:, :, 405:471, 401:601])

    output = torch.nn.functional.interpolate(output, image.shape[-2:], mode='bilinear', align_corners=True)
    writer.add_image('Train.3.Ours', colorize(vutils.make_grid(output.data, nrow=6, normalize=False)), epoch)
    writer.add_image('Train.3.Diff',
                     colorize(vutils.make_grid(torch.abs(output - depth_n).data, nrow=6, normalize=False)), epoch)
    del image
    del depth_n
    del output


if __name__ == '__main__':
    main()
