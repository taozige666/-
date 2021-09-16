import torch
from math import exp
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.loss import chamfer_distance
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################################

class SILogLoss(nn.Module):  # Main loss function used in AdaBins paper
    def __init__(self):
        super(SILogLoss, self).__init__()
        self.name = 'SILog'

    def forward(self, input, target, mask=None, interpolate=True):
        if interpolate:
            input = nn.functional.interpolate(input, target.shape[-2:], mode='bilinear', align_corners=True)

        if mask is not None:
            input = input[mask]
            target = target[mask]
        g = torch.log(input) - torch.log(target)

        Dg = torch.mean(torch.pow(g, 2)) - 0.5 * torch.pow(torch.mean(g), 2)
        return 10 * torch.sqrt(Dg)
        #return Dg


class BinsChamferLoss(nn.Module):  # Bin centers regularizer used in AdaBins paper
    def __init__(self):
        super().__init__()
        self.name = "ChamferLoss"

    def forward(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1
        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        # dist1, dist2, idx1, idx2 = chamfer_distance(input_points,target_points)
        # loss = (torch.mean(dist1)) + (torch.mean(dist2))
        # print("ChamferLoss", loss)
        # loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        # loss = torch.tensor(loss.item())
        return loss

###########################################################################################
def edge_conv2d(im):
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 卷积输出通道
    sobel_kernel = np.repeat(sobel_kernel, 1, axis=1)
    # 输入图的通道
    sobel_kernel = np.repeat(sobel_kernel, 1, axis=0)

    conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()

    edge_detect = conv_op(im)
    return edge_detect


class BoundaryLoss(nn.Module):
  def __init__(self):
        super(BoundaryLoss, self).__init__()
        self.name = 'BoundaryLoss'

  def forward(self, pred, target):
        pred_edge = edge_conv2d(pred)
        target_edge = edge_conv2d(target)


        # edge_detect = np.transpose(pred_edge, (1, 2, 0))
        
        # plt.imshow('edge.jpg', edge_detect)
        #cv2.waitKey(0)
        # plt.show()
        # l_boundary = nn.MSELoss()
        l_boundary = nn.L1Loss()
        # l_boundary = nn.SmoothL1Loss()
        loss = l_boundary(pred_edge, target_edge)
        ''' from matplotlib import pyplot as plt
        pred_edge = pred_edge.squeeze().detach().cpu().numpy()
        plt.subplot(121), plt.imshow(pred_edge)
        target_edge = target_edge.squeeze().detach().cpu().numpy()
        plt.subplot(122), plt.imshow(target_edge)
        plt.show()'''
        return loss

        ######################################################################

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        self.name = 'GradientLoss'
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).cuda()

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).cuda()

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)


    def forward(self, pred, target):
        grad_x = F.conv2d(pred, self.weight_x)
        grad_y = F.conv2d(pred, self.weight_y)
        t_grad_x = F.conv2d(target, self.weight_x)
        t_grad_y = F.conv2d(target, self.weight_y)

        '''gradient = torch.abs(grad_x) + torch.abs(grad_y)
        t_gradient = torch.abs(t_grad_x) + torch.abs(t_grad_y)
        l_gradient = nn.L1Loss()
        loss = l_gradient(gradient, t_gradient)'''

        l_gradient = nn.L1Loss()
        x_loss = l_gradient(grad_x, t_grad_x)
        y_loss = l_gradient(grad_y, t_grad_y)
        loss = x_loss + y_loss
        loss = torch.sqrt(loss)
        return 10 * loss
