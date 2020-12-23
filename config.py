from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from easydict import EasyDict as edict


cfg = edict()


# whether us gpu
cfg.name  = "chairs-experiment"
cfg.path_to_data  = "chairs_train"
cfg.path_to_test_data  = "chairs_val"
cfg.batch_size  = 8
cfg.lr  = 2e-4
cfg.epochs  = 60
cfg.ssim_loss_weight  = 0.1
cfg.save_freq  = 1
cfg.img_shape  = [3, 128, 128]
cfg.channels_2d  = [64, 64, 128, 128, 128, 128, 256, 256, 128, 128, 128]
cfg.strides_2d  = [1, 1, 2, 1, 2, 1, 2, 1, -2, 1, 1]
cfg.channels_3d  = [32, 32, 128, 128, 128, 64, 64, 64]
cfg.strides_3d  = [1, 1, 2, 1, 1, -2, 1, 1]
cfg.num_channels_projection  = [512, 256, 256]
cfg.num_channels_inv_projection  = [256, 512, 1024]
cfg.mode  = "bilinear"



