import numpy as np
import argparse

from model.ViT_adaptive_bins import ViTAdaBins
import torch
from dataloader import getTestData
from torchvision import transforms


def DepthNorm(depth, maxDepth=1000):
    return maxDepth / depth


def write_depth(path, depth, vmin=10, vmax=1000, bits=2):
    depth = depth.numpy()[0, :, :]
    # vmin = depth.min()
    # vmax = depth.max()
    '''max_val = (2 ** (8 * bits)) - 1
            if vmin != vmax:
                 out = (depth - vmin) / (vmax - vmin)
            else:
                out = depth * 0
            # print(out[203:236, 201:301])
            print(out.min(), out.max())'''
    out = depth
    print(out.min(), out.max())
    import cv2
    if bits == 1:
        cv2.imwrite(path, (out).astype("uint8"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif bits == 2:
        cv2.imwrite(path, out.astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return


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


def save_image(i, tpye):
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.set_size_inches(6.4 / 3, 4.8 / 3)  # dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    fig.savefig('/home/liu/test_images/{0}_{1}.png'.format(i, tpye), format='png', transparent=True, dpi=300,
                pad_inches=0)


parser = argparse.ArgumentParser(description='ViT and Adaptive Bins for Monocular Depth Estimation')
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                    choices=['linear', 'softmax', 'sigmoid'])
parser.add_argument('--n_bins', default=256, type=int, help='number of bins to divide depth range')
parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', default=False, help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

args = parser.parse_args()

device = torch.device('cuda')

# create model
model = ViTAdaBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth, norm=args.norm).to(
    device)
checkpoint = torch.load(
    '/home/liu/MyDPTAdaModel/runs/Jun22_20-16-13_3090VisionTransformer_2-lr0.0001-e30-bs2/epoch4_best_rmse_params.pt')
# checkpoint = checkpoint['model']
model.load_state_dict(checkpoint)
model = model.eval()
print('Model created.')

# load data
test_loader = getTestData(batch_size=args.bs)

depth_scores = np.zeros((6, len(test_loader)))  # six metrics
with torch.no_grad():
    for i, sample_batched in enumerate(test_loader):

        model.eval()
        # Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda())
        depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        import torchvision

        print(image)
        torchvision.utils.save_image(image, '/home/liu/test_images/test/rgb_{0}.png'.format(i))
        # cv2.imwrite('/home/liu/test_images/test/rgb_{0}.png'.format(i), image.squeeze(0).permute(1, 2, 0).cpu().numpy()*10000)
        # Normalize depth
        depth_n = depth.cpu().numpy() / 1000
        # Predict
        pred = model(image)[-1].cpu().numpy()
        pred = DepthNorm(pred) / 100
        from skimage import transform

        # pred = transform.resize(pred, (1, 1, 480, 640))
        # pred = np.clip(DepthNorm(pred, maxDepth=1000), 10, 1000) / 100
        # print(depth_n.shape, output.shape)
        # print(depth_n.shape)
        # print('depth_n', depth_n[:, :, 203:236, 201:301], 'pred', pred[:, :,203:236, 201:301])
        flip_trans = transforms.RandomHorizontalFlip(p=1)
        image_flip = flip_trans(image)
        pred_flip = model(image_flip)[-1].cpu().numpy()
        pred_flip = DepthNorm(pred_flip) / 100
        # pred_flip = transform.resize(pred_flip, (1, 1, 480, 640))
        # pred_flip = np.clip(DepthNorm(pred_flip, maxDepth=1000), 10, 1000) / 100

        '''import matplotlib.pyplot as plt
        import torchvision.utils as vutils
        from utils import colorize 
        true_y = DepthNorm(depth_n)
        pred_y = DepthNorm(pred_flip)
        output = torch.tensor(true_y).squeeze(0).cpu()
        output = colorize(vutils.make_grid(output.data, nrow=6, normalize=False))
        output = torch.tensor(output).permute(1, 2, 0).cpu().numpy()
        plt.subplot(1, 2, 1), plt.imshow(output)
        output1 = torch.tensor(pred_y).squeeze(0).cpu()
        output1 = colorize(vutils.make_grid(output1.data, nrow=6, normalize=False))
        output1 = torch.tensor(output1).permute(1, 2, 0).cpu().numpy()
        plt.subplot(1, 2, 2), plt.imshow(output1)
        plt.show()'''

        # valid_mask = np.ones(depth_n.shape)
        # print(pred.min(), pred.max(), pred.mean())
        valid_mask = np.logical_and(depth_n > 0.001, depth_n < 10)

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
        '''print(final.shape)
        print(depth_n.shape)
        print(eval_mask.shape)'''
        errors = compute_errors(depth_n[valid_mask], final[valid_mask])
        print(i)
        print(errors[4])

        ###############################
        import torchvision

        png = (torch.Tensor(final).squeeze(0)).cpu() * 1000
        gt = (torch.Tensor(depth_n)).squeeze(0).cpu() * 1000
        # print(png.shape)
        # write_depth('/home/liu/test_images/test/pred_{0}.png'.format(i),png)
        '''if i ==501:
            print(final[:, :,201:250, 401:450])
            break'''
        # write_depth('/home/liu/test_images/test/gt_{0}.png'.format(i),gt)
        # print('depth_n', torch.Tensor(gt)[:, :, 203:236, 201:301], 'pred', torch.Tensor(png)[:, :,203:236, 201:301])
        '''if errors[4] > 1:
            a = np.array(valid_mask)
            a = a * 1000
            # print(a)
            import matplotlib.pyplot as plt
            import torchvision.utils as vutils
            from utils import colorize 
            true_y = DepthNorm(depth_n)
            pred_y = DepthNorm(final)
            mask = torch.Tensor(a)
            output = torch.tensor(true_y).squeeze(0).cpu()
            output = colorize(vutils.make_grid(output.data, nrow=6, normalize=False))
            output = torch.tensor(output).permute(1, 2, 0).cpu().numpy()
            # plt.subplot(2, 2, 1), 
            plt.imshow(output)
            save_image(i, 'gt')
            output1 = torch.tensor(pred_y).squeeze(0).cpu()
            output1 = colorize(vutils.make_grid(output1.data, nrow=6, normalize=False))
            output1 = torch.tensor(output1).permute(1, 2, 0).cpu().numpy()
            # plt.subplot(2, 2, 2), 
            plt.imshow(output1)
            save_image(i, 'pred')
            output2 = torch.tensor(mask).squeeze(0).cpu()
            output2 = colorize(vutils.make_grid(output2.data, nrow=6, normalize=False))
            output2 = torch.tensor(output2).permute(1, 2, 0).cpu().numpy()
            # plt.subplot(2, 2, 3), 
            plt.imshow(output2)
            img = image.squeeze().permute(1, 2, 0).cpu().numpy()
            # plt.subplot(2, 2, 4), 
            plt.imshow(img)
            save_image(i, 'rgb')
            # plt.show()'''
        for k in range(len(errors)):
            depth_scores[k][i] = errors[k]

e = depth_scores.mean(axis=1)

print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5]))
