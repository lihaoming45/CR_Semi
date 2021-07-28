import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import make_onehot_torch


def point_sample(input, point_coords, **kwargs):
    """
    Args:
        input (Tensor): A tensor of shape (N, C, D, H, W) that contains features map on a D x H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 3) or (N, Dgrid, Hgrid, Wgrid, 3) that contains
        [0, 1] x [0, 1] normalized point coordinates.
    """

    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2).unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3).squeeze(3)
    return output


@torch.no_grad()
def sampling_points(mask, N, k=3, beta=0.75, training=True):
    """
        Follows 3.1. Point Selection for Inference and Training
        In Train:, `The sampling strategy selects N points on a feature map to train on.`
        In Inference, `then selects the N most uncertain points`
        Args:
            mask(Tensor): [B, C, D, H, W]
            N(int): `During training we sample as many points as there are on a stride 16 feature map of the input`
            k(int): Over generation multiplier
            beta(float): ratio of importance points
            training(bool): flag
        Return:
            selected_point(Tensor) : flattened indexing points [B, num_points, 2]
        """

    assert mask.dim() == 5, "Dim must be N(Batch)CDHW"
    device = mask.device
    B, _, D, H, W = mask.shape
    mask, _ = mask.sort(1, descending=True)

    if not training:
        D_step, H_step, W_step = 1 / D, 1 / H, 1 / W
        N = min(D * H * W, N)
        uncertainty_map = -1 * (mask[:, 0] - mask[:, 1])
        _, idx = uncertainty_map.view(B, -1).topk(N, dim=1)  # idx:N个最不确定的点的索引

        points = torch.zeros(B, N, 3, dtype=torch.float, device=device)
        # 因为推理时候没用使用rand生成[0,1]的随机索引，所以需要从idx反推回去属于[0,1]的索引
        points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step  # 表示列索引号
        points[:, :, 1] = H_step / 2.0 + (idx % (H * W) // W).to(torch.float) * H_step  # 表示行索引号
        points[:, :, 2] = D_step / 2.0 + (idx // (H * W)).to(torch.float) * D_step  # 表示层索引号

        return idx, points

    # Official Comment : point_features.py#92
    # It is crucial to calculate uncertanty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to worse results. To illustrate the difference: a sampled point between two coarse predictions
    # with -1 and 1 logits has 0 logit prediction and therefore 0 uncertainty value, however, if one
    # calculates uncertainties for the coarse predictions first (-1 and -1) and sampe it for the
    # center point, they will get -1 unceratinty.

    over_generation = torch.rand(B, k * N, 3, device=device)  # 随机选取一堆点的坐标
    over_generation_map = point_sample(mask, over_generation,
                                       align_corners=False)  # 获取mask上这些坐标位置上的值（可以理解为概率值）(N,C,P) C为类别数

    uncertainty_map = -1 * (over_generation_map[:, 0] - over_generation_map[:, 1])
    _, idx = uncertainty_map.topk(int(beta * N), -1)  # idx:over_generation 上的最重要的beta*N个点的位置

    shift = (k * N) * torch.arange(B, dtype=torch.long, device=device)

    idx += shift[:, None]  # batch_size=1的话相当于没变

    importance = over_generation.view(-1, 3)[idx.view(-1), :].view(B, int(beta * N), 3)
    coverage = torch.rand(B, N - int(beta * N), 3, device=device)
    return torch.cat([importance, coverage], 1).to(device)


def points_mapping(idx, N, coarse, p2_1, p2_2, device):
    B, C, D, H, W = coarse.shape
    D_step, H_step, W_step = 1 / D, 1 / H, 1 / W
    points = torch.zeros(B, N, 3, dtype=torch.float, device=device)

    points[:, :, 0] = W_step / 2.0 + (idx % W).to(torch.float) * W_step  # 表示列索引号
    points[:, :, 1] = H_step / 2.0 + (idx % (H * W) // W).to(torch.float) * H_step  # 表示行索引号
    points[:, :, 2] = D_step / 2.0 + (idx // (H * W)).to(torch.float) * D_step  # 表示层索引号

    coarse = point_sample(coarse, points, align_corners=False)
    fine_1 = point_sample(p2_1, points, align_corners=False)
    fine_2 = point_sample(p2_2, points, align_corners=False)

    return torch.cat([coarse, fine_1, fine_2], dim=1)


def cossim_fn2(x1, x2):
    B = x1.shape[0]
    cos_sim_dim2 = nn.CosineSimilarity(dim=2)

    x2 = x2.view(1, B, x2.shape[-1])
    out = cos_sim_dim2(x1.view(B, 1, x1.shape[-1]), x2)
    out = out.view(out.shape[0], -1)
    return out


def cl_lossfc(features, temperature=0.5, eps=0.000001, is_F=True, is_O=False):
    features = features.squeeze()
    assert features.dim() == 2
    features = features.permute(1, 0)
    Oh_v, Ol_v, Fh_v, Fl_v = torch.chunk(features, 4, 0)
    loss = 0
    if is_F:
        pos_F = cossim_fn2(Fh_v, Fl_v)
        neg_F = cossim_fn2(Fh_v, Oh_v)
        pos_F = torch.exp(pos_F)
        neg_F = torch.exp(neg_F)
        temp_F = (torch.log(pos_F.sum(dim=-1) / (neg_F.sum(dim=-1) + eps)) + 2) / 4
        loss_F = (1 - temp_F).mean()
        # loss_F = (1 - torch.log(pos_F.sum(dim=-1) / (neg_F.sum(dim=-1) + eps))).mean()
        loss += loss_F

    if is_O:
        pos_O = cossim_fn2(Oh_v, Ol_v)
        neg_O = cossim_fn2(Oh_v, Fh_v)
        pos_O = torch.exp(pos_O)
        neg_O = torch.exp(neg_O)
        temp_O = (torch.log(pos_O.sum(dim=-1) / (neg_O.sum(dim=-1) + eps)) + 2) / 4
        loss_O = (1 - temp_O).mean()
        # loss_O = (1 - torch.log(pos_O.sum(dim=-1) / (neg_O.sum(dim=-1) + eps))).mean()
        loss += loss_O

    if is_O and is_F:
        loss /= 2

    return loss


def constrastive_learning(coarse, p2_1, p2_2, gt_cr, N=256, is_F=True, is_O=False):
    assert coarse.dim() == 5, "Dim must be N(Batch)CDHW"
    assert coarse.size()[-3:] == gt_cr.size()[-3:]

    device = coarse.device
    B, C, D, H, W = coarse.shape
    # D_step, H_step, W_step = 1 / D, 1 / H, 1 / W

    coarse = coarse.softmax(1)
    gtF = ((gt_cr == 1) * 1).view(B, -1).int()
    gtO = ((gt_cr != 0) * 1).view(B, -1).int()

    probO_map = coarse[:, 2, ...].view(B, -1)
    probF_map = coarse[:, 1, ...].view(B, -1)


    # 卵巢 高/低置信度的mask
    assert probO_map.size() == gtO.size()
    probO_maskH = torch.ge(probO_map, 0.9).int() * gtO
    probO_maskL = torch.ge(probO_map, 0.3).int() * torch.le(probO_map, 0.8).int() * gtO

    # 卵泡 高/低置信度的mask
    assert probF_map.size() == gtF.size()
    probF_maskH = torch.ge(probF_map, 0.9).int() * gtF
    probF_maskL = torch.ge(probF_map, 0.3).int() * torch.le(probF_map, 0.8).int() * gtF

    numSlect_list = [probO_maskH.sum(), probO_maskL.sum(), probF_maskH.sum(), probF_maskL.sum()]

    if min(numSlect_list) < N:
        return 0
        # N = min(numSlect_list)
    elif min(numSlect_list) > 1024:
        N = 1024

    # if N == 0:
    #     return 0

    random_mask = torch.rand_like(probF_maskL.float())

    _, Oh_idx = (probO_maskH.half() - random_mask.half()).topk(N)
    _, Ol_idx = (probO_maskL.half() - random_mask.half()).topk(N)
    _, Fh_idx = (probF_maskH.half() - random_mask.half()).topk(N)
    _, Fl_idx = (probF_maskL.half() - random_mask.half()).topk(N)

    Oh_feature = points_mapping(Oh_idx, N, coarse, p2_1, p2_2, device)  # (B,C,N) B=1
    Ol_feature = points_mapping(Ol_idx, N, coarse, p2_1, p2_2, device)
    Fh_feature = points_mapping(Fh_idx, N, coarse, p2_1, p2_2, device)
    Fl_feature = points_mapping(Fl_idx, N, coarse, p2_1, p2_2, device)

    cl_loss = cl_lossfc(torch.cat([Oh_feature, Ol_feature, Fh_feature, Fl_feature], dim=-1), is_F=is_F, is_O=is_O)
    return cl_loss
