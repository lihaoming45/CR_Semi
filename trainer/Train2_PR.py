import os
from dataop.dataop import MySet, create_list
from torch.utils.data import DataLoader
# from model.Unet_3D_V3 import UNet
# from model.Modified3DUnet import Modified3DUNet
from model.PointRend2.Unet_3D_PR2 import Unet_3D
from model.PointRend2.pointRend2 import PointRend, PointHead
from model.PointRend2.sampling_point2 import point_sample
from model.PointRend_3D import Point_Selection
import torch
from torch import optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
import logging
import click
from utils import eval_metric, dice_focal_loss, patch_splice, dice_point_loss
from tensorboardX import SummaryWriter
# from dataop_patches import train_val_loader #这个函数，不进行3d resize 直接扣
from model.PointRend.MY_GE_gen import train_val_loader  # spatial size进行了resize，再扣patch
from model.PointRend.MY_GE_gen import create_list_MYGE
from apex import amp


def build_network(snapshot):
    epoch = 0
    # net = UNet()
    # net = Modified3DUNet()
    net = PointRend(Unet_3D(n_channels=1, n_classes=3), PointHead()).cuda()

    # net = nn.DataParallel(net)
    if snapshot is not None:
        _, _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    # net = net.cuda()
    return net, epoch


@click.command()
@click.option('--data-path', type=str, default="/home/lhm/All_data/formal_data_3D(MY_GE)",
              help='Path to dataset folder')
# @click.option('--unlabel-data-path', type=str, default="/home/lhm/All_data/formal_data_3D(MY_GE)/Unlabeled_data",
#               help='Path to unlabeled dataset folder')
@click.option('--models-path', type=str, default="/home/lhm/3D_Unet/model/PointRend2/snapshots/snapshot_Train2",
              help='Path for storing model snapshots')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--batch-size', type=int, default=1)
@click.option('--epochs', type=int, default=150, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='2', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.0001)
@click.option('--num_class', type=str, default=3, help='number target of classification  ')
@click.option('--patch_decision', type=bool, default=True, help='decision for using patches')
@click.option('--patch_shape', type=tuple, default=(128, 192, 192), help='shape of every patch')
@click.option('--stride_shape', type=tuple, default=(64, 192, 192), help='stride of patch')
@click.option('--new_shape', type=tuple, default=(192, 192), help='resize shape of spatial')
def train(data_path, test_path, models_path, snapshot, batch_size, epochs, start_lr, milestones, gpu,
          num_class, patch_decision, patch_shape, stride_shape, new_shape):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # torch.cuda.set_device(0)
    net, starting_epoch = build_network(snapshot)

    data_path = os.path.abspath(os.path.expanduser(data_path))
    models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)
    # train_list, test_list = create_list(data_path, ratio=0.9)

    if patch_decision:
        train_list, test_list = create_list_MYGE(data_path, split_list=['GE_all','MY'], ratio=0.8, data_folder='data_Crop',
                                                 label_folder='label_Crop',crop_vaild=False, is_small=False)

        train_set, test_set = train_val_loader(train_list, test_list, patch_shape, stride_shape, new_shape,
                                               aug_decision=False)
        max_steps_train = train_set.cumulative_sizes[-1]
        max_steps_test = test_set.cumulative_sizes[-1]


    else:
        train_list, test_list = create_list(data_path, ratio=0.9)
        train_set = MySet(train_list, num_class=3, phase='train', aug_decision=False)
        test_set = MySet(test_list, num_class=3, phase='val', aug_decision=True)

        max_steps_train = train_set.__len__()
        max_steps_test = test_set.__len__()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=start_lr)
    # optimizer_Rend = optim.Adam(point_net.parameters(), lr=0.01)
    # scheduler_rend = ReduceLROnPlateau(optimizer_Rend, 'min', factor=0.5, patience=5, verbose=True)

    # optimizer = optim.SGD(net.parameters(), lr=start_lr, momentum=0.9)
    # scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')])
    # scheduler2 = CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-4)
    # scheduler3 = CosineAnnealingLR(optimizer,T_max=20,eta_min=1e-4)

    writer = SummaryWriter(log_dir='logs_tensorboardX_Train2', comment='loss')

    # seg_criterion = dice_loss(weight=False, num_class=num_class)
    # seg_criterion = dice_focal_loss(weight=False, num_class=num_class)
    # point_criterion = dice_point_loss(weight=False, num_class=num_class)
    # class_criterion = nn.CrossEntropyLoss()

    net, optimizer = amp.initialize(net, optimizer, opt_level="O1", verbosity=0)
    for epoch in range(starting_epoch, starting_epoch + epochs):

        trainEpo_eval = {i: [] for i in range(2)}
        testEpo_eval = {i: [] for i in range(6)}
        epoch_losses = []
        epoch_val_losses = []
        count_eval_acc_list = []
        epochval_loss_list = [20000]

        train_iterator = tqdm(train_loader, total=max_steps_train // batch_size + 1)
        net.train()
        net.istraining = True
        for x, y, idx, key_word in train_iterator:

            x, y = x.to(0, non_blocking=True), y.to(0, non_blocking=True)
            result = net(x)
            pred = F.interpolate(result["coarse"], x.shape[-3:], mode="trilinear", align_corners=True)

            seg_loss = F.cross_entropy(pred, y.long())

            gt_points = point_sample(y.float().unsqueeze(1), result["points"], mode="nearest",
                                     align_corners=False).squeeze_(1).long()
            # points_loss = point_criterion(F.softmax(result["rend"], dim=1), gt_points, device=gt_points.device)
            points_loss = F.cross_entropy(result["rend"], gt_points)

            loss = seg_loss + points_loss
            epoch_losses.append(loss.item())

            trainEval = eval_metric(pred, y, num_class=num_class)
            for i in range(trainEval.__len__()):
                trainEpo_eval[i].append(trainEval[i])

            status = 'id:{} // TRAIN_epoch={} --- loss={:.4f} --- epoch_loss={:.4f}  --- diceF={:.4f} --- diceO={:.4f}' \
                .format(idx,
                        epoch + 1, loss.item(), np.mean(epoch_losses), np.mean(trainEpo_eval[0]),
                        np.mean(trainEpo_eval[1]))

            train_iterator.set_description(status)

            optimizer.zero_grad()
            # loss.backward()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
        train_iterator.close()

        writer.add_scalar('scalar/loss_train', np.mean(epoch_losses), epoch)
        writer.add_scalar('scalar/DiceF_train', np.mean(trainEpo_eval[0]), epoch)
        writer.add_scalar('scalar/DiceO_train', np.mean(trainEpo_eval[1]), epoch)

        if epoch % 5 == 0:
            out_patch = []
            out_rend_patch = []

            status_list = []
            statusR_list = []
            net.eval()
            with torch.no_grad():
                test_iteraror = tqdm(test_loader, total=max_steps_test // batch_size + 1)
                for x, y, idx, patch_num, label_vol, data_shape, key_word,ora_shape in test_iteraror:
                    # x, y = Variable(x).cuda(), Variable(y).cuda()
                    x, y = x.to(0, non_blocking=True), y.to(0, non_blocking=True)
                    patch_num = patch_num.cpu().numpy()

                    result = net(x)

                    fine_pred = result['fine'] # 输出没经过softmax
                    coarse_pred = F.interpolate(result["coarse"], x.shape[-3:], mode="trilinear", align_corners=True)# 输出没经过softmax


                    seg_loss = F.cross_entropy(coarse_pred, y.long())
                    loss = seg_loss
                    epoch_val_losses.append(loss.item())

                    if patch_decision:
                        patch_idx = idx.cpu().numpy() + 1
                        # 存储一个体数据的所有小切块
                        out_patch.append(coarse_pred.data.cpu().numpy())
                        out_rend_patch.append(fine_pred.data.cpu().numpy())

                        if patch_idx[0] % patch_num == 0:
                            # data_shape = [x.cpu().numpy() for x in data_shape]
                            splice_pred_vol = patch_splice(out_patch, num_class, data_shape, patch_shape, stride_shape)
                            splice_Rpred_vol = patch_splice(out_rend_patch, num_class, data_shape, patch_shape,
                                                            stride_shape)

                            eval_splice_list = eval_metric(splice_pred_vol, label_vol, num_class=num_class,
                                                           patch_decision=True, is_training=False)
                            evalR_splice_list = eval_metric(splice_Rpred_vol, label_vol, num_class=num_class,
                                                            patch_decision=True, is_training=False)

                            # status_splice = 'spliceEval:Test_epoch={} -- bg={:.3f} -- foll={:.4f} -- Ova={:.4f} --- OvaAll={:.4f}'.format(
                            #     epoch + 1, evalDice_splice_dict[0], evalDice_splice_dict[1],
                            #     evalDice_splice_dict[2], evalDice_splice_dict[3])
                            status_list.append(eval_splice_list)
                            statusR_list.append(evalR_splice_list)

                            out_patch.clear()
                            out_rend_patch.clear()

                    evalDice_dict = eval_metric(coarse_pred, y, num_class=num_class, is_training=False)
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

            # min_loss = min(epochval_loss_list)
            # epochval_loss_list.append(np.mean(epochval_loss_list))

            # scheduler.step()
            if np.mean(testEpo_eval[0]) > 0.80:
                torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["UnetPR_3D", str(epoch + 1)])))
                # if epoch >= 50:
                #     torch.save(point_net.state_dict(),
                #                os.path.join(models_path, "snapshots_MLP", '_'.join(["MLP", str(epoch + 1)])))

    writer.close()

if __name__ == '__main__':
    train()
