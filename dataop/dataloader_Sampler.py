import os
import torch
import numpy as np
import torch.nn.functional as F
import itertools
from dataop import ramps
from torch.utils.data.sampler import Sampler


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size=2, secondary_batch_size=1):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


def uncertainty_eval(unlabeled_x, ema_net):
    T = 8
    volume_batch_r = unlabeled_x.repeat(2, 1, 1, 1, 1)
    stride = volume_batch_r.shape[0] // 2
    preds = torch.zeros([stride * T, 3, 128, 192, 192]).cuda()
    for i in range(T // 2):
        ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
        with torch.no_grad():
            preds[2 * stride * i:2 * stride * (i + 1)] = ema_net(ema_inputs)

    preds = F.softmax(preds, dim=1)
    preds = preds.reshape(T, stride, 3, 128, 192, 192)
    preds = torch.mean(preds, dim=0)  # (batch, 3, 128,192,192)
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # (batch, 1, 128,192,192)
    return uncertainty


def uncertainty_eval_CRSim(unlabeled_x, ema_net, shape):
    T = 8
    volume_batch_r = unlabeled_x.repeat(2, 1, 1, 1, 1)
    stride = volume_batch_r.shape[0] // 2
    preds = torch.zeros([stride * T, 3, shape[0], shape[1], shape[2]]).cuda()
    for i in range(T // 2):
        ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
        with torch.no_grad():
            ema_out = ema_net(ema_inputs)  # 只包含coarse和res2
            ema_pred = F.interpolate(ema_out["coarse"], preds.shape[-3:], mode="trilinear", align_corners=False)
            preds[2 * stride * i:2 * stride * (i + 1)] = ema_pred

    preds = F.softmax(preds, dim=1)
    preds = preds.reshape(T, stride, 3, shape[0], shape[1], shape[2])
    preds = torch.mean(preds, dim=0)  # (batch, 3, 128,192,192)
    uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)  # (batch, 1, 128,192,192)
    return uncertainty


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 0.1 * ramps.sigmoid_rampup(epoch, 40.0)


def update_ema_variables(model, ema_model, alpha, global_step,muti_gpu=True):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if muti_gpu:
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data.to(1))
        else:
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss
