import os
from torch.utils.data import DataLoader
from models import Unet_3D_aspp, Unet_3D, asym_bakcbone, Rend_CR
from utils.sampling_point2 import point_sample,constrastive_learning
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import logging
from utils.utils import dice_loss, eval_metric, dice_focal_loss, patch_splice, judge_fn
from tensorboardX import SummaryWriter
from dataop.generator import train_val_loader
from dataop.generator import create_list_MYGE
from apex import amp


def build_network(snapshot,cfg):
    epoch = 0
    # net = UNet()
    # net = Modified3DUNet()

    if cfg.model_name is 'unet':
        net = Unet_3D.Unet_3D(n_channels=1, n_classes=3)
    elif cfg.model_name is 'aspp':
        net = Unet_3D_aspp.Unet_3D_aspp(n_channels=1, n_classes=3)
    elif cfg.model_name in ['cr','pr']:
        net = Rend_CR.Rend_m(asym_bakcbone(n_channels=1, n_classes=3), Rend_CR.Render_h()).cuda()

    if snapshot is not None:
        _, _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    # net = net.cuda()
    return net, epoch


def train(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    net, starting_epoch = build_network(cfg.snapshot)
    data_path = os.path.abspath(os.path.expanduser(cfg.data_path))
    models_path = os.path.abspath(os.path.expanduser(cfg.models_path))
    os.makedirs(models_path, exist_ok=True)

    train_list, test_list = create_list_MYGE(data_path, split_list=['GE_all', 'MY'], ratio=0.8,
                                             data_folder='data_Crop',
                                             label_folder='label_Crop', crop_vaild=False, is_small=False)

    train_set, test_set = train_val_loader(train_list, test_list, cfg.patch_shape, cfg.stride_shape, cfg.new_shape,
                                           aug_decision=False)
    max_steps_train = train_set.cumulative_sizes[-1]
    max_steps_test = test_set.cumulative_sizes[-1]

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=cfg.lr)
    writer = SummaryWriter(log_dir='logs/logs_tensorboardX', comment='loss')

    net, optimizer = amp.initialize(net, optimizer, opt_level="O1", verbosity=0)
    for epoch in range(starting_epoch, starting_epoch + cfg.epochs):
        trainEpo_eval = {i: [] for i in range(2)}
        testEpo_eval = {i: [] for i in range(6)}
        epoch_losses = []
        epoch_val_losses = []
        epo_clloss = [0.]

        train_iterator = tqdm(train_loader, total=max_steps_train // cfg.batch_size + 1)
        net.train()
        for x, y,y_cr, idx in train_iterator:
            optimizer.zero_grad()
            x, y, y_cr = x.to(0, non_blocking=True), y.to(0, non_blocking=True), y_cr.to(0, non_blocking=True)

            out = net(x)

            if cfg.model_name in ['unet','aspp']:

                # ----------------segloss设置----------------
                seg_crossloss = F.cross_entropy(out, y.long())
                # ----------------segloss设置----------------
                loss = seg_crossloss
                epoch_losses.append(loss.item())

                trainEval = eval_metric(out, y, num_class=cfg.num_class)  # including: diceF,diceO


            elif cfg.model_name in ['cr','pr']:
                pred = F.interpolate(out["coarse"], x.shape[-3:], mode="trilinear", align_corners=True)
                seg_loss = F.cross_entropy(pred, y.long())
                gt_points = point_sample(y.float().unsqueeze(1), out["points"], mode="nearest",
                                         align_corners=False).squeeze_(1).long()

                points_loss = F.cross_entropy(out["rend"], gt_points)
                loss = seg_loss + points_loss

                if cfg.model_name is 'cr':
                    if epoch > 12:
                        cl_loss = constrastive_learning(out["coarse"], out['p2_1'], out['p2_2'], y_cr,
                                                        is_F=True,
                                                        is_O=True)
                        if torch.is_tensor(cl_loss):
                            loss = loss + 0.2 * cl_loss
                            epo_clloss.append(cl_loss.item())
                trainEval = eval_metric(pred, y, num_class=cfg.num_class)

            for i in range(trainEval.__len__()):
                trainEpo_eval[i].append(trainEval[i])

            status = 'id:{} // TRAIN_epoch={} --- loss={:.4f} --- epoch_loss={:.4f}  --- diceF={:.4f} --- diceO={:.4f}' \
                .format(idx,
                        epoch + 1, loss.item(), np.mean(epoch_losses), np.mean(trainEpo_eval[0]),
                        np.mean(trainEpo_eval[1]))

            train_iterator.set_description(status)
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        train_iterator.close()

        writer.add_scalar('scalar/scalars_train', np.mean(epoch_losses), epoch)
        if epoch % cfg.log_every_epoch == 0:
            out_patch = []
            status_list = []
            net.eval()
            test_iterator = tqdm(test_loader, total=max_steps_test // cfg.batch_size + 1)
            with torch.no_grad():
                for x, y, idx, patch_num, label_vol, data_shape,v_spacing in test_iterator:
                    x, y = x.to(0, non_blocking=True), y.to(0, non_blocking=True)
                    patch_num = patch_num.cpu().numpy()
                    out = net(x)

                    if cfg.model_name in ['cr','pr']:
                        out = out['fine']

                    # ----------------segloss setting----------------
                    seg_loss = F.cross_entropy(out, y.long())
                    # ----------------segloss setting----------------

                    loss = seg_loss
                    epoch_val_losses.append(loss.item())

                    patch_idx = idx.cpu().numpy() + 1
                    out_patch.append(out.data.cpu().numpy())
                    if patch_idx[0] % patch_num == 0:
                        splice_pred_vol = patch_splice(out_patch, cfg.num_class, data_shape, cfg.patch_shape, cfg.stride_shape)
                        evalDice_splice_list = eval_metric(splice_pred_vol, label_vol, num_class=cfg.num_class,
                                                           patch_decision=True, is_training=False)

                        status_list.append(evalDice_splice_list)
                        out_patch.clear()

                    evalDice_dict = eval_metric(out, y, num_class=cfg.num_class, is_training=False)
                    for i in range(6):
                        testEpo_eval[i].append(evalDice_dict[i])

                    status = 'Test_epo={} -- epoch_loss={:.4f} -- diceF={:.4f} -- diceO={:.4f} -- hdF={:.3f} -- hdO={:.3f} -- asdF={:.3f} -- asdO={:.3f}' \
                        .format(
                        epoch + 1, np.mean(epoch_val_losses), np.mean(testEpo_eval[0]),
                        np.mean(testEpo_eval[1]), np.mean(testEpo_eval[2]), np.mean(testEpo_eval[3]),
                        np.mean(testEpo_eval[4]), np.mean(testEpo_eval[5]))

                    test_iterator.set_description(status)


                test_iterator.close()
            mean_status_list = np.mean(status_list, axis=0)
            description = 'spliceEval:Test_epoch={} -- diceF={:.4f} -- diceO={:.4f} -- hdF={:.3f} -- hdO={:.3f} -- asdF={:.3f} -- asdO={:.3f}'.format(
                epoch + 1, mean_status_list[0], mean_status_list[1], mean_status_list[2], mean_status_list[3],
                mean_status_list[4], mean_status_list[5])
            print(description)


            if np.mean(testEpo_eval[0]) > 0.86:
                torch.save(net.state_dict(), os.path.join(models_path, '_'.join([cfg.model_name, str(epoch + 1)])))

    writer.close()



if __name__ == '__main__':
    train()
