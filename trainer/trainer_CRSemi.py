import os
from torch.utils.data import DataLoader

from models.asym_bakcbone import asym_model
from models.PointRend_CRSemi import PointRend, PointHead
from utils.sampling_point2 import point_sample, constrastive_learning
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from utils.utils import eval_metric, patch_splice
from tensorboardX import SummaryWriter
from dataop.generator4_CRSemi import train_val_loader,create_list_MYGE
from dataop.dataloader_Sampler import TwoStreamBatchSampler, uncertainty_eval_CRSim, get_current_consistency_weight, \
    softmax_mse_loss, update_ema_variables
from dataop import ramps
from apex import amp


def build_network(snapshot):

    net = PointRend(asym_model(n_channels=1, n_classes=3), PointHead())
    epoch = 0
    if snapshot is not None:
        _, _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()

    return net, epoch

def build_emaNetwork():
        net = asym_model(n_channels=1, n_classes=3)
        net.eval()
        # net.use_checkpoint=False
        for param in net.parameters():
            param.detach_()
        net = net.cuda()
        return net


def train(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    # torch.cuda.set_device(0)
    net, starting_epoch = build_network(cfg.snapshot)
    ema_net= build_emaNetwork()

    data_path = os.path.abspath(os.path.expanduser(cfg.data_path))
    models_path = os.path.abspath(os.path.expanduser(cfg.models_path))
    os.makedirs(models_path, exist_ok=True)
    # train_list, test_list = create_list(data_path, ratio=0.9)



    train_list, test_list, unlabel_list = create_list_MYGE(data_path,cfg.unlabel_data_path, split_list=['GE_all', 'MY'], ratio=0.8,
                                                 data_folder='data_Crop',
                                                 label_folder='label_Crop', crop_vaild=False, is_small=False)


    train_set, test_set,cut_indx = train_val_loader(train_list, test_list, unlabel_list, cfg.patch_shape, cfg.stride_shape,
                                           cfg.new_shape,
                                           aug_decision=False)
    max_steps_train = train_set.cumulative_sizes[-1]


    labeled_idxs = list(range(cut_indx))
    unlabeled_idxs = list(range(cut_indx, max_steps_train - 1))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, cfg.batch_size, 1)



    train_loader = DataLoader(train_set, batch_sampler=batch_sampler, num_workers=4, pin_memory=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size-1, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=cfg.start_lr)

    writer = SummaryWriter(log_dir='logs_tensorboardX_Train4_CRSemi', comment='loss')
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1", verbosity=0)

    max_iterations = (cfg.epochs - 1) * len(train_loader)

    for epoch in range(starting_epoch, starting_epoch + cfg.epochs):

        trainEpo_eval = {i: [] for i in range(2)}
        testEpo_eval = {i: [] for i in range(6)}
        epoch_losses = []
        epo_clloss = [0.]
        epoch_val_losses = []

        train_iterator = tqdm(train_loader)
        net.train()
        net.istraining = True
        for x, y, y_cr, idx in train_iterator:
            assert len(y[1:, ...].unique()) == 1
            assert len(y_cr[1:, ...].unique()) == 1
            x, y = x.to(0, non_blocking=True), y[:1, ...].to(0, non_blocking=True)
            y_cr = y_cr[:1, ...].to(0, non_blocking=True)

            unlabeled_x = x[1:]
            noise = torch.clamp(torch.randn_like(unlabeled_x) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_x + noise

            ema_coarse = ema_net(ema_inputs)['coarse']
            ema_pred = F.interpolate(ema_coarse, x.shape[-3:], mode="trilinear", align_corners=False)

            result = net(x)
            pred = F.interpolate(result["coarse"], x.shape[-3:], mode="trilinear", align_corners=True)

            seg_loss = F.cross_entropy(pred[:1, ...], y.long())

            # ----------------emaloss setting-----------------
            uncertainty = uncertainty_eval_CRSim(unlabeled_x, ema_net,x.shape[-3:])

            consistency_weight = get_current_consistency_weight(epoch // cfg.epochs)
            consistency_dist = softmax_mse_loss(pred[1:, ...], ema_pred)  # (batch, 3, 128,192,192)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(epoch, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            consistency_dist = torch.sum(mask * consistency_dist) / (2 * torch.sum(mask) + 1e-10)
            consistency_loss = consistency_weight * consistency_dist


            gt_points = point_sample(y.float().unsqueeze(1), result["points"], mode="nearest",
                                     align_corners=False).squeeze_(1).long()
            points_loss = F.cross_entropy(result["rend"], gt_points)
            loss = seg_loss + points_loss + consistency_loss

            if epoch>=16:
                cl_loss = constrastive_learning(result["coarse"][:1,...], result['p2_1'][:1,...], result['p2_2'][:1,...], y_cr, is_F=True,
                                                is_O=True)
                if torch.is_tensor(cl_loss):
                    loss = loss+0.2*cl_loss
                    epo_clloss.append(cl_loss.item())


            epoch_losses.append(loss.item())

            trainEval = eval_metric(pred, y, num_class=cfg.num_class)
            for i in range(trainEval.__len__()):
                trainEpo_eval[i].append(trainEval[i])

            status = 'id:{} // TRAIN_epoch={} --- cl_loss={:.4f} --- epoch_loss={:.4f}  --- diceF={:.4f} --- diceO={:.4f}' \
                .format(idx,
                        epoch + 1, np.mean(epo_clloss), np.mean(epoch_losses), np.mean(trainEpo_eval[0]),
                        np.mean(trainEpo_eval[1]))

            train_iterator.set_description(status)

            optimizer.zero_grad()

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            update_ema_variables(net.backbone, ema_net, 0.99, epoch,muti_gpu=False)

        train_iterator.close()

        writer.add_scalar('scalar/loss_train', np.mean(epoch_losses), epoch)
        writer.add_scalar('scalar/DiceF_train', np.mean(trainEpo_eval[0]), epoch)
        writer.add_scalar('scalar/DiceO_train', np.mean(trainEpo_eval[1]), epoch)

        if epoch % cfg.log_every_epoch == 0:
            out_patch = []
            out_rend_patch = []
            status_list = []
            statusR_list = []
            net.eval()
            with torch.no_grad():
                test_iteraror = tqdm(test_loader)
                for x, y, idx, patch_num, label_vol, data_shape, ora_shape in test_iteraror:
                    x, y = x.to(0, non_blocking=True), y.to(0, non_blocking=True)
                    patch_num = patch_num.cpu().numpy()
                    result = net(x)

                    fine_pred = result['fine']  # 输出没经过softmax
                    coarse_pred = F.interpolate(result["coarse"], x.shape[-3:], mode="trilinear",
                                                align_corners=True)  # 输出没经过softmax

                    seg_loss = F.cross_entropy(coarse_pred, y.long())
                    loss = seg_loss
                    epoch_val_losses.append(loss.item())


                    patch_idx = idx.cpu().numpy() + 1
                    # 存储一个体数据的所有小切块
                    out_patch.append(coarse_pred.data.cpu().numpy())
                    out_rend_patch.append(fine_pred.data.cpu().numpy())

                    if patch_idx[0] % patch_num == 0:
                        # data_shape = [x.cpu().numpy() for x in data_shape]
                        splice_pred_vol = patch_splice(out_patch, cfg.num_class, data_shape, cfg.patch_shape, cfg.stride_shape)
                        splice_Rpred_vol = patch_splice(out_rend_patch, cfg.num_class, data_shape, cfg.patch_shape,
                                                        cfg.stride_shape)

                        eval_splice_list = eval_metric(splice_pred_vol, label_vol, num_class=cfg.num_class,
                                                       patch_decision=True, is_training=False)
                        evalR_splice_list = eval_metric(splice_Rpred_vol, label_vol, num_class=cfg.num_class,
                                                        patch_decision=True, is_training=False)

                        status_list.append(eval_splice_list)
                        statusR_list.append(evalR_splice_list)

                        out_patch.clear()
                        out_rend_patch.clear()

                    evalDice_dict = eval_metric(coarse_pred, y, num_class=cfg.num_class, is_training=False)
                    for i in range(6):
                        testEpo_eval[i].append(evalDice_dict[i])

                    status = 'Test_epo={} -- epoch_loss={:.4f} -- diceF={:.4f} -- diceO={:.4f} -- hdF={:.3f} -- hdO={:.3f} -- asdF={:.3f} -- asdO={:.3f}' \
                        .format(
                        epoch + 1, np.mean(epoch_val_losses), np.mean(testEpo_eval[0]),
                        np.mean(testEpo_eval[1]), np.mean(testEpo_eval[2]), np.mean(testEpo_eval[3]),
                        np.mean(testEpo_eval[4]), np.mean(testEpo_eval[5]))

                    test_iteraror.set_description(status)
            mean_status_list = np.mean(status_list, axis=0)
            mean_statusR_list = np.mean(statusR_list, axis=0)
            description = 'spliEval(O/R):Test_epo{} -- diceF={:.4f}//{:.4f} -- diceO={:.4f}//{:.4f} -- hdF={:.3f}//{:.3f} ' \
                          '-- hdO={:.3f}//{:.3f} -- asdF={:.3f}//{:.3f} -- asdO={:.3f}//{:.3f}'.format(epoch + 1,
                mean_status_list[0], mean_statusR_list[0], mean_status_list[1], mean_statusR_list[1],
                mean_status_list[2], mean_statusR_list[2],mean_status_list[3], mean_statusR_list[3],
                mean_status_list[4], mean_statusR_list[4],mean_status_list[5], mean_statusR_list[5])
            print(description)

            if np.mean(testEpo_eval[0]) > 0.87:
                torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["Unet_3D_CRSemi", str(epoch + 1)])))

    writer.close()


if __name__ == '__main__':
    from utils.cfg_parser import Config
    default_cfg_path = 'configs/configs/CR_Semi.yaml'
    cfg = {}
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    train(cfg)
