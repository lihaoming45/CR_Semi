import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from hdf5 import SliceBuilder
from skimage import morphology
from scipy.ndimage import measurements, filters, binary_dilation, binary_erosion
from model.PointRend_3D import _if_near, getpoint
import math
from binary import hd, asd, jc
import SimpleITK as sitk
from resize_3D import itk_resample
import nibabel as nib
import os
import xlsxwriter
import copy


# slice_bulider = SliceBuilder(np.zeros(shape=(288, 224, 256)), np.zeros(shape=(288, 224, 256)),
#                              patch_shape=(144, 112, 128), stride_shape=(96, 56, 64))
# slices = slice_bulider.raw_slices


gray_list = [0, 128, 255]


class AvgMeter(object):
    """
    Acc meter class, use the update to add the current acc
    and self.avg to get the avg acc
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_one_hot(data, num_class):
    one_hot_list = []
    for i in range(num_class):
        temp = np.zeros_like(data, dtype=np.float32)
        index = np.where(data == float(i))
        temp[index] = 1.
        one_hot_list.append(temp)
    one_hot = np.array(one_hot_list, dtype=np.float32)
    one_hot = np.transpose(one_hot, (1, 0, 2, 3, 4))
    return one_hot


def make_one_hot_pr(data, num_class):
    one_hot_list = []
    for i in range(num_class):
        temp = np.zeros_like(data, dtype=np.float32)
        index = np.where(data == float(i))
        temp[index] = 1.
        one_hot_list.append(temp)
    one_hot = np.array(one_hot_list, dtype=np.float32)
    one_hot = np.transpose(one_hot, (1, 0, 2))
    return one_hot


def getpoint(mask_img, k=3, beta=0.70, training=True, nearest_neighbor=2, threshold=[0.4, 0.6]):
    if training:
        d, w, h = mask_img.shape

    else:
        cc, d, w, h = mask_img.shape
    N = int(beta * k * w * h * d)
    dxy_min = [0, 0, 0]
    dxy_max = [d - 1, w - 1, h - 1]
    points = np.random.uniform(low=dxy_min, high=dxy_max, size=(N, 3))

    # print(points)
    if (beta > 1 or beta < 0):
        print("beta should be in range [0,1]")
        return None

    # for the training, the mask is a hard mask
    if training == True:
        if beta == 0: return points
        res = []
        mask_blank = np.zeros_like(mask_img)
        for p in points:
            if (_if_near(p, mask_img, nearest_neighbor)) and tuple(np.uint8(p)) not in res:
                res.append(tuple(np.uint8(p)))
                mask_blank[tuple(np.uint8(p))] = 1

        others = int((1 - beta) * k * w * h)
        not_edge_points = np.random.uniform(low=dxy_min, high=dxy_max, size=(others, 3))
        for p in not_edge_points:
            if tuple(np.uint8(p)) not in res:
                res.append(tuple(np.uint8(p)))
                mask_blank[tuple(np.uint8(p))] = 1
        return res, mask_blank


def point_selection(coarse_pred, up_shape=(32, 56, 56)):
    assert len(coarse_pred.size()) == 4
    coarse_pred = torch.argmax(coarse_pred, dim=0)
    res_point, point_mask = getpoint(mask_img=coarse_pred)
    point_mask = F.upsample(torch.from_numpy(point_mask).unsqueeze(0).unsqueeze(0).float(), size=up_shape,
                            mode='trilinear', align_corners=True)
    point_mask = point_mask.squeeze()
    point_mask = torch.where(
        (point_mask >= 0.9) | ((point_mask > 0.4) & (point_mask < 0.5)) | ((point_mask > 0.7) & (point_mask < 0.8)),
        torch.ones_like(point_mask), torch.zeros_like(point_mask))

    return point_mask.bool()


smooth = 0.000001


class dice_loss2(nn.Module):
    def __init__(self, weight, num_class):
        super(dice_loss2, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def forward(self, pred, truth, weight=None):
        N = truth.size(0)
        truth = truth.cpu().numpy()
        truth = make_one_hot(truth, num_class=self.num_class)

        y_true = torch.from_numpy(truth).cuda()

        if self.weight:
            weight = torch.from_numpy(weight).cuda()
            y_true = y_true * weight
            pred = pred * weight

        background_pred = pred[:, 0, :, :, :]
        background_true = y_true[:, 0, :, :, :]

        foll_pred = pred[:, 1, :, :, :]
        foll_true = y_true[:, 1, :, :, :]

        ova_pred = pred[:, 1, :, :, :] + pred[:, 2, :, :, :]
        ova_true = y_true[:, 1, :, :, :] + y_true[:, 2, :, :, :]

        background_pred = background_pred.view(N, -1)
        background_true = background_true.view(N, -1)

        foll_pred = foll_pred.view(N, -1)
        foll_true = foll_true.view(N, -1)

        ova_pred = ova_pred.view(N, -1)
        ova_true = ova_true.view(N, -1)

        # y_true = y_true.view(N, -1)
        # y_pred = pred.view(N, -1)

        intersection_bg = torch.sum(background_pred * background_true)
        intersection_foll = torch.sum(foll_pred * foll_true)
        intersection_ova = torch.sum(ova_pred * ova_true)

        loss_bg = -torch.log(2.0 * intersection_bg + smooth) + \
                  torch.log(torch.sum(background_true) + torch.sum(background_pred) + smooth)

        loss_foll = -torch.log(2.0 * intersection_foll + smooth) + \
                    torch.log(torch.sum(foll_true) + torch.sum(foll_pred) + smooth)

        loss_ova = -torch.log(2.0 * intersection_ova + smooth) + \
                   torch.log(torch.sum(ova_true) + torch.sum(ova_pred) + smooth)
        # intersection = torch.sum(y_true * y_pred)
        # loss = -torch.log(2.0 * intersection + smooth) + \
        #        torch.log(torch.sum(y_true) + torch.sum(y_pred) + smooth)

        loss = (0.1 * loss_bg + loss_foll + 0.6 * loss_ova) / 1.7
        return loss


class dice_loss(nn.Module):
    def __init__(self, weight, num_class):
        super(dice_loss, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def forward(self, pred, truth, weight=None):
        N = truth.size(0)
        truth = truth.cpu().numpy()
        truth = make_one_hot(truth, num_class=self.num_class)

        y_true = torch.from_numpy(truth).cuda()

        if self.weight:
            weight = torch.from_numpy(weight).cuda()
            y_true = y_true * weight
            pred = pred * weight

        background_pred = pred[:, 0, :, :, :]
        background_true = y_true[:, 0, :, :, :]

        res_pred = pred[:, 1:, :, :, :]
        res_true = y_true[:, 1:, :, :, :]

        background_pred = background_pred.view(N, -1)
        background_true = background_true.view(N, -1)

        res_pred = res_pred.view(N, -1)
        res_true = res_true.view(N, -1)

        # y_true = y_true.view(N, -1)
        # y_pred = pred.view(N, -1)

        intersection_bg = torch.sum(background_pred * background_true)
        intersection_res = torch.sum(res_pred * res_true)

        loss_bg = -torch.log(2.0 * intersection_bg + smooth) + \
                  torch.log(torch.sum(background_true) + torch.sum(background_pred) + smooth)

        loss_res = -torch.log(2.0 * intersection_res + smooth) + \
                   torch.log(torch.sum(res_true) + torch.sum(res_pred) + smooth)

        # intersection = torch.sum(y_true * y_pred)
        # loss = -torch.log(2.0 * intersection + smooth) + \
        #        torch.log(torch.sum(y_true) + torch.sum(y_pred) + smooth)

        # loss = (0.4* loss_bg + loss_res)/1.4
        loss = 0.9*loss_res+0.1*loss_bg

        return loss


def eval_metric(pred_vol, label_vol, num_class, patch_decision=False, is_training=True, add_jc=False,
                voxel_spacing=None):
    truth = label_vol.data.cpu().numpy()
    if patch_decision:
        pred_vol = pred_vol
    else:
        pred_vol = pred_vol.data.cpu().numpy()

    truth = make_one_hot(truth, num_class=num_class)
    if len(pred_vol.shape) == 5:
        pred_vol = np.argmax(pred_vol, axis=1)



    predO = (pred_vol != 0) * 1
    truthO = truth[:, 1, :, :, :] + truth[:, 2, :, :, :]

    predF = (pred_vol == 1) * 1

    truthF = truth[:, 1, :, :, :]
    diceO = (2 * (predO * truthO).sum() + 0.000001) / (truthO.sum() + predO.sum() + 0.000001)
    diceF = (2 * (predF * truthF).sum() + 0.000001) / (truthF.sum() + predF.sum() + 0.000001)
    if is_training:
        # evalmetric_dic = {'diceF': diceF, 'diceO': diceO}
        return [diceF, diceO]

    # if voxel_spacing is not None:
    if np.sum(predF) != 0:
        hdF = hd(np.squeeze(predF), np.squeeze(truthF), voxelspacing=voxel_spacing)
        asdF = asd(np.squeeze(predF), np.squeeze(truthF), voxelspacing=voxel_spacing)
    else:
        hdF = None
        asdF = None

    if np.sum(predO) != 0:
        hdO = hd(np.squeeze(predO), np.squeeze(truthO), voxelspacing=voxel_spacing)
        asdO = asd(np.squeeze(predO), np.squeeze(truthO), voxelspacing=voxel_spacing)
    else:
        hdO = None
        asdO = None

    # else:
    #     hdO = hd(predO, truthO, voxelspacing=voxel_spacing)
    #     hdF = hd(predF, truthF, voxelspacing=voxel_spacing)
    #     asdO = asd(predO, truthO, voxelspacing=voxel_spacing)
    #     asdF = asd(predF, truthF, voxelspacing=voxel_spacing)
    # evalmetric_dic = {'diceF': diceF, 'diceO': diceO, 'hdF': hdF, 'hdO': hdO, 'asdF': asdF, 'asdO': asdO}

    if add_jc:
        jcF = jc(predF, truthF)
        jcO = jc(predO, truthO)

        return [diceF, diceO, hdF, hdO, asdF, asdO, jcF, jcO]

    return [diceF, diceO, hdF, hdO, asdF, asdO]

    # for i in range(num_class):
    #     pred = (pred_vol[0, :, :, :] == i) * 1
    #     pred = pred.flatten()
    #     label = truth[0, i, :, :, :].flatten()
    #     dice = (2 * (pred * label).sum() + 0.000001) / (label.sum() + pred.sum() + 0.000001)
    #
    #     # evalDice_dict[i] = dice
    #     evalDice_list.append(dice)
    # if num_class > 2:
    #     pred_OvaAll = (pred_vol != 0) * 1
    #     pred_OvaAll = pred_OvaAll.flatten()
    #     true_OvaAll = truth[:, 1, :, :, :] + truth[:, 2, :, :, :]
    #     true_OvaAll = true_OvaAll.flatten()
    #     Dice_OvaAll = (2 * (pred_OvaAll * true_OvaAll).sum() + 0.000001) / (
    #             true_OvaAll.sum() + pred_OvaAll.sum() + 0.000001)
    #
    #     # evalDice_dict[i + 1] = Dice_OvaAll
    #     evalDice_list.append(Dice_OvaAll)

    # return evalDice_list


def evaluate_Dice(pred_vol, label_vol):
    evalDice_dict = {}

    truth = label_vol.data.cpu().numpy()
    pred_vol = pred_vol.data.cpu().numpy()

    classes_num = len(np.unique(truth))

    truth = make_one_hot(truth, num_class=classes_num)
    pred_vol = np.argmax(pred_vol, axis=1)

    for i in range(classes_num):
        pred = (pred_vol[0, :, :, :] == i) * 1
        pred = pred.flatten()
        label = truth[0, i, :, :, :].flatten()
        dice = (2 * (pred * label).sum() + 0.000001) / (label.sum() + pred.sum() + 0.000001)

        evalDice_dict[i] = dice

    return evalDice_dict


class dice_focal_loss(nn.Module):
    def __init__(self, weight, num_class):
        super(dice_focal_loss, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def dice_coef(self, y_true, y_pred, device, smooth=1e-5):
        alpha = 0.5
        beta = 0.5
        # y_true = tf.one_hot(y_true, depth=3)
        # y_true = make_onehot_torch(y_true,num_class=self.num_class)#Tensor形式的onehot

        # y_true = y_true.cpu().numpy()
        # y_true = make_one_hot(y_true, num_class=self.num_class)
        # y_true = torch.from_numpy(y_true).cuda(device)

        # ones = tf.ones(tf.shape(y_true))
        ones = torch.ones(y_true.shape).cuda(device)
        p0 = y_pred
        p1 = ones - y_pred
        g0 = y_true
        g1 = ones - y_true
        # num = tf.reduce_sum(p0 * g0, axis=[0, 1, 2])
        # num = torch.sum(p0 * g0,dim=0)
        num = (p0 * g0).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True).sum(dim=4,
                                                                                                       keepdim=True).squeeze()

        # den = num + alpha * tf.reduce_sum(p0 * g1, axis=[0, 1, 2]) + beta * tf.reduce_sum(p1 * g0, axis=[0, 1, 2])
        # den = num + alpha * torch.sum(p0 * g1, dim=(0,2,3)) + beta *torch.sum(p1 * g0, dim=(0,2,3))
        den = num + alpha * (p0 * g1).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True).sum(
            dim=4, keepdim=True).squeeze() + \
              beta * (p1 * g0).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True).sum(dim=4,
                                                                                                              keepdim=True).squeeze()

        return torch.sum(num / ((den + smooth) * 3.))

    def dice_loss(self, y_true, y_pred, device):
        return 1 - self.dice_coef(y_true, y_pred, device=device)

    def focal_loss(self, y_true, y_pred, device):
        gamma = 2.
        alpha = .1
        # y_true = tf.to_int32(y_true)
        # y_true = tf.one_hot(y_true, depth=3)
        # torch.cuda.synchronize()
        # y_true = make_onehot_torch(y_true,num_class=self.num_class)#Tensor形式的onehot

        # y_true = y_true.cpu().numpy()
        # y_true = make_one_hot(y_true, num_class=self.num_class)
        # y_true = torch.from_numpy(y_true).cuda(device)
        # torch.cuda.synchronize()
        # ones = torch.ones_like(y_pred)
        p1 = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        p0 = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
        p1 = torch.clamp(p1, 1e-5, .99999)
        p0 = torch.clamp(p0, 1e-5, .99999)
        return (-torch.sum((1 - alpha) * torch.log(p1)) - torch.sum(
            alpha * torch.pow(p0, gamma) * torch.log(1. - p0))) / 3.

    def forward(self, pred, truth, device):
        beta = 0.02
        # truth[truth>=3]=0
        # truth[truth<=0]=0

        truth = truth.cpu().numpy()
        truth = make_one_hot(truth, num_class=self.num_class)
        truth = torch.from_numpy(truth).cuda(device)
        # truth = truth.cpu().numpy()

        return beta * self.focal_loss(truth, pred, device) + torch.log(self.dice_loss(truth, pred, device))


class dice_point_loss(nn.Module):
    def __init__(self, weight, num_class):
        super(dice_point_loss, self).__init__()
        self.weight = weight
        self.num_class = num_class

    def dice_coef(self, y_true, y_pred, device, smooth=1e-5):
        alpha = 0.5
        beta = 0.5
        # assert y_pred.size()[0]==point_mask.sum()

        # y_true = y_true.cpu().numpy()
        # y_true = make_one_hot(y_true, num_class=self.num_class)
        # y_true = torch.from_numpy(y_true).cuda(device)
        #
        # y_true = torch.masked_select(y_true, (point_mask * torch.ones_like(y_true)).bool())
        # y_true = y_true.reshape((point_mask.sum(), y_true.size()[1]))
        # assert y_pred.size()==y_true.size()

        ones = torch.ones(y_true.shape).cuda(device)
        p0 = y_pred
        p1 = ones - y_pred
        g0 = y_true
        g1 = ones - y_true
        # num = tf.reduce_sum(p0 * g0, axis=[0, 1, 2])
        # num = torch.sum(p0 * g0,dim=0)
        num = (p0 * g0).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).squeeze()

        den = num + alpha * (p0 * g1).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True).squeeze() + beta * (
                p1 * g0).sum(dim=0,
                             keepdim=True).sum(dim=2, keepdim=True).squeeze()

        return torch.sum(num / ((den + smooth) * 3.))

    def dice_loss(self, y_true, y_pred, device):
        return 1 - self.dice_coef(y_true, y_pred, device=device)

    def focal_loss(self, y_true, y_pred, device):
        gamma = 2.
        alpha = .1
        # y_true = tf.to_int32(y_true)
        # y_true = tf.one_hot(y_true, depth=3)
        # y_true = y_true.cpu().numpy()
        # y_true = make_one_hot(y_true, num_class=self.num_class)
        # y_true = torch.from_numpy(y_true).cuda(device)

        # ones = torch.ones_like(y_pred)
        p1 = torch.where(y_true == 1, y_pred, torch.ones_like(y_pred))
        p0 = torch.where(y_true == 0, y_pred, torch.zeros_like(y_pred))
        p1 = torch.clamp(p1, 1e-5, .99999)
        p0 = torch.clamp(p0, 1e-5, .99999)
        return (-torch.sum((1 - alpha) * torch.log(p1)) - torch.sum(
            alpha * torch.pow(p0, gamma) * torch.log(1. - p0))) / 3.

    def forward(self, pred, truth, device, point_mask=None):
        beta = 0.02
        truth = truth.cpu().numpy()
        truth = make_one_hot_pr(truth, num_class=self.num_class)
        truth = torch.from_numpy(truth).cuda(device)

        # truth = torch.masked_select(truth, (point_mask.float() * torch.ones_like(truth)).bool())
        # truth = truth.reshape((point_mask.sum(), self.num_class))

        assert pred.size() == truth.size()

        return beta * self.focal_loss(truth, pred, device) + torch.log(self.dice_loss(truth, pred, device))


# patch拼接函数
def patch_splice(patchPred_list, num_class, data_shape, patch_shape, stride_shape):
    # patchPred_list=list(map(lambda x:np.squeeze(x),patchPred_list))
    data_shape = tuple([x.cpu().numpy()[0] for x in data_shape])

    slice_bulider = SliceBuilder(np.zeros(shape=data_shape), np.zeros(shape=data_shape),
                                 patch_shape=patch_shape, stride_shape=stride_shape)
    slices = slice_bulider.raw_slices

    blank_vol = np.zeros(shape=(1, num_class,) + data_shape)
    blank_count = np.zeros_like(blank_vol)

    assert len(patchPred_list) == len(slices)
    for patch_pred, silce_idx in zip(patchPred_list, slices):
        channel_slice = (slice(0, 1), slice(0, num_class))
        silce_idx = channel_slice + tuple(silce_idx)
        blank_vol[silce_idx] += patch_pred
        blank_count[silce_idx] += 1

    prediction_maps = blank_vol / blank_count  # shape=(1,3,288,224,256),得到整体volume的概率图

    return prediction_maps





def patch_spliceMask(patchMask_list, data_shape, patch_shape, stride_shape):
    data_shape = tuple([x.cpu().numpy()[0] for x in data_shape])

    slice_bulider = SliceBuilder(np.zeros(shape=data_shape), np.zeros(shape=data_shape),
                                 patch_shape=patch_shape, stride_shape=stride_shape)
    slices = slice_bulider.raw_slices

    blank_vol = np.zeros(data_shape)
    blank_count = np.zeros_like(blank_vol)

    for patch_pred, silce_idx in zip(patchMask_list, slices):
        silce_idx = tuple(silce_idx)
        blank_vol[silce_idx] += patch_pred
        blank_count[silce_idx] += 1
    mask_vol = blank_vol / blank_count
    mask_vol = np.where(mask_vol >= 0.5, 1, 0)
    return mask_vol


# 拼接后Dice评估函数
def eval_splice(pred_classlist, true):
    evalDice_dict = {}

    for i, pred_class_vol in enumerate(pred_classlist):
        true_class_vol = (true == i) * 1
        pred_class_vol = (pred_class_vol >= 0.5) * 1

        dice = (2 * (pred_class_vol * true_class_vol).sum() + 0.000001) / (
                true_class_vol.sum() + pred_class_vol.sum() + 0.000001)
        evalDice_dict[i] = dice

    pred_OvaAll = (pred_classlist[1] + pred_classlist[2]) / 2
    true_OvaAll = (true != 0) * 1
    Dice_OvaAll = (2 * (pred_OvaAll * true_OvaAll).sum() + 0.000001) / (
            true_OvaAll.sum() + pred_OvaAll.sum() + 0.000001)
    evalDice_dict[i + 1] = Dice_OvaAll

    return evalDice_dict


def make_onehot_torch(data, num_class):
    # ordering: (N,C,D,H,W)
    N = data.shape[0]
    new_shape = (N,) + (num_class,) + tuple(data.shape[1:])
    data = data.long().unsqueeze(1)
    # torch.cuda.synchronize()
    data = data.cpu()
    onehot = torch.zeros(new_shape).scatter(1, data, 1)

    return onehot.cuda()



def AWM(pred, truth, last_Weight, iter_num, alpha, num_class, begin_epoch):
    pred = pred.detach().cpu().numpy()
    truth = truth.cpu().numpy()
    truth = make_one_hot(truth, num_class=3)
    # Weight_Map_first = np.zeros_like(pred)

    # truth = truth.numpy()
    # truth = make_one_hot(truth, num_class=num_class)

    if iter_num == begin_epoch:
        Weight_Map = np.ones_like(pred)

        return Weight_Map
    else:

        Energy = np.zeros_like(truth)
        for i in range(num_class):
            truth_i = truth[:, i, :, :, :]
            pred_i = pred[:, i, :, :, :]
            Energy[:, i, :, :, :] = np.where(truth_i == 1., pred_i, (1 - pred_i))

        Weight_Map = (1 - Energy) + alpha * Energy * last_Weight

        return Weight_Map


def foll_count(mask, thresh_max=20, stride=5, one_hot_mode=False):
    num_classes = thresh_max // stride + 1
    one_hot = np.zeros((num_classes, 1))

    foll_mask = (mask == 1) * 1
    _, num_foll = measurements.label(foll_mask)
    id = math.floor(num_foll / stride)
    if id > num_classes - 1:
        id = num_classes - 1

    one_hot[id] = 1
    if one_hot_mode:
        return one_hot
    else:
        return id


def edge_mask(mask, kernel_size=2, muti_class=False):
    kernel = morphology.ball(kernel_size)

    ova_mask = (mask != 0) * 1
    foll_mask = (mask == 1) * 1
    # _, num_foll = measurements.label(foll_mask)
    foll_edge = binary_dilation(foll_mask == 0, kernel) * foll_mask
    ova_edge = binary_dilation(ova_mask == 0, kernel) * ova_mask * 2

    Edge_mask = foll_edge + ova_edge
    Edge_mask[Edge_mask == 3] = 2
    assert len(np.unique(Edge_mask)) == 3

    return Edge_mask


def edge_pred(pred, kernel_size=2):
    pred = pred.data.cpu().numpy()

    mask = np.argmax(pred, axis=1)

    batch_store = np.zeros_like(mask)
    for i in range(pred.shape[0]):
        batch_store[i, ...] = edge_mask(mask[i, ...], kernel_size=kernel_size)

    boundary_mask = batch_store
    bg_pred = bg_mask(pred, boundary_mask)

    pred_mask = make_one_hot(boundary_mask, num_class=3)

    pred_edge = pred_mask * pred
    pred_edge[:, 0, ...] = bg_pred

    return torch.from_numpy(pred_edge).cuda()


def bg_mask(pred, boundary_mask):
    mask = np.argmax(pred, axis=1)
    mask = (mask != 0) * 1
    boundary_mask = (boundary_mask != 0) * 1

    bg_new_mask = mask - boundary_mask  # 表明是除去边界里面空心部分的mask
    pred_bg_new = (1 - pred[:, 0, ...]) * bg_new_mask

    bg_pred = pred[:, 0, ...] + pred_bg_new

    return bg_pred


def judge_fn(y_count, out_class):
    out_class = np.array(out_class)
    if y_count == out_class:
        return 1
    else:
        return 0


def aug_fn(self, trans_builder, data):
    mean, std = self.calculate_mean_std(data)
    trans_builder.mean = mean
    trans_builder.std = std
    trans_builder.phase = self.phase
    transformer = trans_builder.build()

    data_transformer = transformer.raw_transform()
    data = data_transformer(data)

    return data


def SimCLR_loss(out1, out2, temperature=0.5, eps=0.000001, B=2):
    # if (len(out1.size()), len(out2.size())) != (2, 2):
    #     a=1
    assert len(out1.size()) == 2 and len(out2.size()) == 2
    # [2B,C]
    out = torch.cat([out1, out2], dim=0)
    # [2B,2B]
    sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * B, device=sim_matrix.device)).bool()

    # [2B,2B-1]
    sim_matrix = sim_matrix.masked_select(mask).view(2 * B, -1)

    # compute_loss
    pos_sim = torch.exp(torch.sum(out1 * out2, dim=-1) / temperature)
    # [2B]
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = (-torch.log(pos_sim / (sim_matrix.sum(dim=-1) + eps))).mean()

    return loss


def Simcos_loss(out1, out2, temperature=0.5, eps=0.000001, B=2):
    cos_sim_dim1 = nn.CosineSimilarity(dim=1)
    cos_sim_dim2 = nn.CosineSimilarity(dim=2)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    negative_mask = torch.ones((B, 2 * B), dtype=bool)
    for i in range(B):
        negative_mask[i, i] = 0
        negative_mask[i, i + B] = 0

    l_pos = cos_sim_dim1(out1, out2).view(B, 1)
    l_pos /= temperature

    negatives = torch.cat([out1, out2], dim=0)

    loss = 0
    labels = torch.zeros(B, dtype=torch.long).cuda()
    for pos in [out1, out2]:
        negatives = negatives.view(1, 2 * B, out1.shape[-1])
        l_neg = cos_sim_dim2(pos.view(B, 1, out1.shape[-1]), negatives)
        l_neg = l_neg[negative_mask].view(l_neg.shape[0], -1)
        l_neg /= temperature

        logits = torch.cat([l_pos, l_neg], dim=1)  # [N,K+1]
        loss += criterion(logits, labels)

    loss = loss / (2 * B)

    return loss


def deal_with_E(out, out_E):
    assert len(out.shape) == 5
    ovaAll_pred = out[:, 2, ...] + out[:, 1, ...]
    ovaAll_E_pred = out_E[:, 2, ...]
    ova_mask = (ovaAll_pred >= 0.5) * (ovaAll_E_pred <= 0.5) * 2

    foll_pred = out[:, 1, ...]
    foll_E_pred = out_E[:, 1, ...]
    foll_mask = (foll_pred >= 0.5) * (foll_E_pred <= 0.5) * 1

    m_pred = foll_mask + ova_mask
    m_pred[m_pred == 3] = 1

    return m_pred


# def multi_result_printf(out, gt, save_path, is_gt, probF_f=None, probO_F=None, maskF_f=None, maskO_F=None):

def remove_radio_cc(input, num_cc, radio):
    input_mask = copy.deepcopy(input)

    for val in range(1, num_cc + 1):
        if np.sum((input_mask == val) * 1) < radio:
            input_mask[input_mask == val] = 0

    return input_mask


def remove_min_cc(input, num_cc):
    area = 0
    idx = 0
    for cc in range(1, num_cc + 1):
        cc_area = np.sum((input == cc) * 1)
        if cc_area > area:
            area = cc_area
            idx = cc

    return idx


def FD_MD_countError(pred, gt, vox_spacing):
    assert len(vox_spacing) == 3
    assert len(gt.shape) == 3
    assert gt.shape == pred.shape

    # -----卵泡区域--------
    foll_pred = (pred == 1) * 1
    foll_truth = (gt == 1) * 1

    # -----卵巢区域--------
    Ova_pred = (pred != 0) * 1  # 卵巢预测区域

    kernel = morphology.ball(3)
    # --------除去卵巢多余区域 以及卵巢外的卵泡区域---------：
    Ova_pred = binary_erosion(Ova_pred, kernel) * 1
    Ova_pred, num_ova = measurements.label(Ova_pred)

    if num_ova > 1:
        idx = remove_min_cc(Ova_pred, num_ova)
        Ova_pred = (Ova_pred == idx) * 1
        foll_pred = Ova_pred * foll_pred  # 去除卵巢外的卵泡

    # --------腐蚀操作抹去粘连-------
    foll_pred = binary_erosion(foll_pred, kernel) * 1

    # ----------卵泡数量检测标记---------
    foll_GT, numGT_CC = measurements.label(foll_truth)
    foll_PR, numPR_CC = measurements.label(foll_pred)
    foll_PR = remove_radio_cc(foll_PR, numPR_CC, radio=100)
    follPR_uni = np.unique(foll_PR)
    numPR_CC = len(follPR_uni) - 1

    # ----------计算直径大于5mm的卵泡Dice-----
    beta = vox_spacing.prod()
    radio = 8180 / beta

    # ------------卵泡FD,MD数量计算-------------
    FD_num, MD_num = 0, 0
    for pr in range(1, numPR_CC + 1):
        id_pr = follPR_uni[pr]
        foll_pr = (foll_PR == id_pr) * 1
        # 只计算卵泡半径在2.5mm以下的识别率：球体积（4/3）*pi*R^3
        if np.sum(foll_pr) < radio:

            idxs = list(np.unique(foll_pr * foll_GT))
            idxs.remove(0)
            if len(idxs) == 0:
                FD_num += 1

    for gt in range(1, numPR_CC + 1):
        foll_gt = (foll_GT == gt) * 1
        idxs = list(np.unique(foll_gt * foll_PR))
        idxs.remove(0)
        if len(idxs) == 0:
            MD_num += 1

    return [(FD_num + 0.000001) / (numPR_CC + 0.000001), (MD_num + 0.000001) / (numGT_CC + 0.000001),
            numPR_CC, numGT_CC]


def vol_eval(mask, gt, vox_spacing):
    if vox_spacing is not None:
        mask_vol = np.sum(mask) * np.prod(vox_spacing)
        gt_vol = np.sum(gt) * np.prod(vox_spacing)

    else:
        mask_vol = np.sum(mask)
        gt_vol = np.sum(gt)

    return mask_vol, gt_vol


def object_registration(pred_mask, gt_mask):
    pred_mask, num_obj = measurements.label(pred_mask)
    store_lists = []

    if num_obj == 0:
        return None

    for pr_val in range(1, num_obj + 1):
        pred_obj = (pred_mask == pr_val) * 1
        dice_obj = (2 * (pred_obj * gt_mask).sum() + 0.000001) / (gt_mask.sum() + pred_obj.sum() + 0.000001)
        store_lists.append(dice_obj)

    regist_val = store_lists.index(max(store_lists)) + 1

    return (pred_mask == regist_val) * 1


def evalMask_metrics(pred_mask, gt_mask, add_jc=True, add_vol=True,
                     voxel_spacing=None):
    dice = (2 * (pred_mask * gt_mask).sum() + 0.000001) / (gt_mask.sum() + pred_mask.sum() + 0.000001)

    if voxel_spacing is not None:
        if np.sum(pred_mask) != 0:
            hd_val = hd(np.squeeze(pred_mask), np.squeeze(gt_mask), voxelspacing=voxel_spacing)
            asd_val = asd(np.squeeze(pred_mask), np.squeeze(gt_mask), voxelspacing=voxel_spacing)
        else:
            hd_val = None
            asd_val = None


    else:
        hd_val = hd(pred_mask, gt_mask, voxelspacing=voxel_spacing)
        asd_val = asd(pred_mask, gt_mask, voxelspacing=voxel_spacing)
    # evalmetric_dic = {'diceF': diceF, 'diceO': diceO, 'hdF': hdF, 'hdO': hdO, 'asdF': asdF, 'asdO': asdO}

    dists_return = [dice, hd_val, asd_val]
    if add_jc:
        jc_val = jc(pred_mask, gt_mask)
        dists_return += [jc_val]

    if add_vol:
        mask_vol, gt_vol = vol_eval(pred_mask, gt_mask, voxel_spacing)
        dists_return += [mask_vol, gt_vol]

    return dists_return


def multi_result_printf(out_list, gt_list, test_list, folder_dict, base_path, save_folder,epoch, spacing_list=None):
    assert len(out_list) == len(gt_list) == len(test_list) == len(spacing_list)

    if folder_dict['ora_data']:
        for i_dir in test_list:
            data_dir = i_dir['data']
            name = os.path.basename(i_dir['data'])
            data = sitk.ReadImage(data_dir)



            data = itk_resample(data, new_shape=(192, 192), islabel=False)


            ora_path = os.path.join(base_path, 'test_data')
            os.makedirs(ora_path, exist_ok=True)
            sitk.WriteImage(data, os.path.join(ora_path, name))

    evalF_list = []
    evalO_list = []
    FDMD_list = []
    singleF_list = []
    name_list = []

    for pred, gt, i_dir, voxel_spacing in zip(out_list, gt_list, test_list, spacing_list):
        name = os.path.basename(i_dir['data'])
        name_list.append(name)


        pred = torch.from_numpy(pred)
        pred = F.softmax(pred, 1)
        pred = pred.squeeze()
        pred = pred.numpy()
        mask = np.argmax(pred, 0)

        if folder_dict['reprocess']:
            Ova_mask = (mask != 0) * 1  # 卵巢预测区域
            kernel = morphology.ball(3)

            # --------除去卵巢多余区域 以及卵巢外的卵泡区域---------：


            Ova_mask = binary_erosion(Ova_mask, kernel) * 1
            Ova_mask, num_ova = measurements.label(Ova_mask)

            if num_ova > 1:
                idx = remove_min_cc(Ova_mask, num_ova)
                Ova_mask = (Ova_mask == idx) * 1
                Ova_mask = binary_dilation(Ova_mask, kernel) * 1

                mask = Ova_mask * mask

        gt = gt.squeeze().data.cpu().numpy()
        gtF = (gt == 1) * 1
        gtO = (gt != 0) * 1

        maskF = (mask == 1) * 1
        maskO = (mask != 0) * 1

        if folder_dict['singleFoll_eval']:
            registF_mask = object_registration(maskF, gtF)

            evalSingleF_state = evalMask_metrics(registF_mask, gtF, add_jc=True,
                                                 voxel_spacing=voxel_spacing)
            singleF_list.append(evalSingleF_state)

        if folder_dict['xls_save']:
            # eval_state = eval_metric(pred, gt, num_class=3, patch_decision=True, is_training=False, add_jc=True,
            #                          voxel_spacing=voxel_spacing)
            evalO_state = evalMask_metrics(maskO, gtO, add_jc=True, add_vol=True,
                                           voxel_spacing=voxel_spacing)
            evalF_state = evalMask_metrics(maskF, gtF, add_jc=True,
                                           voxel_spacing=voxel_spacing)

            evalO_list.append(evalO_state)
            evalF_list.append(evalF_state)

        if folder_dict['FDMD_save']:
            detection_eval = FD_MD_countError(mask, gt, voxel_spacing)
            FDMD_list.append(detection_eval)

        if folder_dict['gt']:
            gt_itk = sitk.GetImageFromArray(gt)
            gt_path = os.path.join(base_path, 'test_gt')
            os.makedirs(gt_path, exist_ok=True)
            sitk.WriteImage(gt_itk, os.path.join(gt_path, name))

        if folder_dict['fullSeg']:
            seg_mask = np.uint8(mask)
            seg_itk = sitk.GetImageFromArray(seg_mask)

            seg_path = os.path.join(base_path, 'full_seg')
            os.makedirs(seg_path, exist_ok=True)
            sitk.WriteImage(seg_itk, os.path.join(seg_path, 'full_seg' + name))

        if folder_dict['probF']:
            probF = pred[1, ...] * 254
            probF = np.uint8(probF)
            probF_itk = sitk.GetImageFromArray(probF)
            sitk.WriteImage(probF_itk, os.path.join(base_path, save_folder, 'probF', 'probF_' + name))

        if folder_dict['probO']:
            probO = (pred[1, ...] + pred[2, ...]) * 254
            probO = np.uint8(probO)
            probO_itk = sitk.GetImageFromArray(probO)
            sitk.WriteImage(probO_itk, os.path.join(base_path, save_folder, 'probO', 'probO_' + name))

        if folder_dict['maskF']:
            maskF = (mask == 1) * 254
            maskF = np.uint8(maskF)
            maskF_itk = sitk.GetImageFromArray(maskF)
            sitk.WriteImage(maskF_itk, os.path.join(base_path, save_folder, 'maskF', 'maskF_' + name))

        if folder_dict['maskO']:
            maskO = (mask != 0) * 254
            maskO = np.uint8(maskO)

            maskO_itk = sitk.GetImageFromArray(maskO)
            sitk.WriteImage(maskO_itk, os.path.join(base_path, save_folder, 'maskO', 'maskO_' + name))

    if bool(evalO_list):
        headings = ['name', 'diceF', 'jcF', 'hdF', 'asdF', 'diceO', 'jcO', 'hdO', 'asdO', 'prO_vol', 'gtO_vol']
        xls_path = os.path.join(base_path, save_folder)
        os.makedirs(xls_path, exist_ok=True)
        path = os.path.join(xls_path, str(epoch)+'_'+'evalResult.xlsx')
        workbook = xlsxwriter.Workbook(path)

        worksheet = workbook.add_worksheet()
        evalO_arr = np.array(evalO_list)
        evalF_arr = np.array(evalF_list)

        diceF_list, diceO_list = list(evalF_arr[:, 0]), list(evalO_arr[:, 0])
        hdF_list, hdO_list = list(evalF_arr[:, 1]), list(evalO_arr[:, 1])
        asdF_list, asdO_list = list(evalF_arr[:, 2]), list(evalO_arr[:, 2])
        jcF_list, jcO_list = list(evalF_arr[:, 3]), list(evalO_arr[:, 3])
        maskO_vol_list, gtO_vol_list = list(evalO_arr[:, 4]), list(evalO_arr[:, 5])

        assert len(name_list) == len(diceF_list)

        worksheet.write_row('A1', headings)

        worksheet.write_column('A2', name_list)

        worksheet.write_column('B2', diceF_list)
        worksheet.write_column('C2', jcF_list)
        worksheet.write_column('D2', hdF_list)
        worksheet.write_column('E2', asdF_list)

        worksheet.write_column('F2', diceO_list)
        worksheet.write_column('G2', jcO_list)
        worksheet.write_column('H2', hdO_list)
        worksheet.write_column('I2', asdO_list)
        worksheet.write_column('J2', maskO_vol_list)
        worksheet.write_column('K2', gtO_vol_list)

        workbook.close()

    if bool(FDMD_list):
        headings = ['name', 'FD', 'MD', 'pr_count', 'gt_count']
        FDMD_path = os.path.join(base_path, save_folder)
        os.makedirs(FDMD_path, exist_ok=True)
        path = os.path.join(FDMD_path, 'FDMD.xlsx')
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet()

        FDMD_arr = np.array(FDMD_list)
        FD_list, MD_list = list(FDMD_arr[:, 0]), list(FDMD_arr[:, 1])
        prCount_list, gtCount_list = list(FDMD_arr[:, 2]), list(FDMD_arr[:, 3])

        worksheet.write_row('A1', headings)
        worksheet.write_column('A2', name_list)
        worksheet.write_column('B2', FD_list)
        worksheet.write_column('C2', MD_list)
        worksheet.write_column('D2', prCount_list)
        worksheet.write_column('E2', gtCount_list)

        workbook.close()

    if bool(singleF_list):
        headings = ['name', 'diceF', 'jcF', 'hdF', 'asdF', 'prF_vol', 'gtF_vol']

        singleF_path = os.path.join(base_path, save_folder)
        os.makedirs(singleF_path, exist_ok=True)
        path = os.path.join(singleF_path, 'SingleF_xls.xlsx')
        workbook = xlsxwriter.Workbook(path)
        worksheet = workbook.add_worksheet()

        evalF_arr = np.array(singleF_list)

        dice_list, jc_list, hd_list, asd_list, pr_vol_list, gt_vol_list = list(evalF_arr[:, 0]), list(
            evalF_arr[:, 1]), list(evalF_arr[:, 2]), list(evalF_arr[:, 3]), list(evalF_arr[:, 4]), list(evalF_arr[:, 5])

        assert len(name_list) == len(singleF_list)

        worksheet.write_row('A1', headings)

        worksheet.write_column('A2', name_list)

        worksheet.write_column('B2', dice_list)
        worksheet.write_column('C2', jc_list)
        worksheet.write_column('D2', hd_list)
        worksheet.write_column('E2', asd_list)

        worksheet.write_column('F2', pr_vol_list)
        worksheet.write_column('G2', gt_vol_list)

        workbook.close()
