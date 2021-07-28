from utils.cfg_parser import Config
import argparse
from trainer import trainer, trainer_CRSemi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training Setting')
    parser = argparse.ArgumentParser('--model-name', type=str, default='cr_semi',
                                     help='choose the model you want to training(including unet, aspp, pr, cr, cr_semi)')
    parser.add_argument('--data-path', type=str, default="/home/lhm/All_data/formal_data_3D(MY_GE)",
                        help='Path to dataset folder')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--unlabel-data-path', type=str, default=None, help='Path to unlabeled dataset folder')
    parser.add_argument('--models-save-path', type=str, default=None, help='Path for storing model snapshots')
    parser.add_argument('--gpu', type=str, default='2', help='List of GPUs for parallel training, e.g. 0,1,2,3')
    parser.add_argument('--models-save-path', type=str, default=None, help='Path for storing model snapshots')
    parser.add_argument('--num_class', type=str, default=3, help='number target of classification')
    parser.add_argument('--patch_shape', type=tuple, default=(128, 192, 192), help='shape of every patch')
    parser.add_argument('--stride_shape', type=tuple, default=(64, 192, 192), help='stride of patch')
    parser.add_argument('--new_shape', type=tuple, default=(192, 192), help='resize shape of spatial')

    default_cfg_path = 'configs/configs/CR_Semi.yaml'

    args = parser.parse_args()
    cfg = vars(args)
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)

    if cfg.model_name in ['cr','pr','unet', 'aspp']:
        trainer.train(cfg)

    if cfg.model_name is 'cr_semi':
        trainer_CRSemi.train(cfg)
