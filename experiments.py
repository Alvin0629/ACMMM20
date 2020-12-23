import json
import os
import sys
import time
import torch
from tool.dataloaders import dataloader1
from models.neural_renderer import NeuralRenderer
from training.training import Trainer
from config import cfg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Save directory.
timestamp = time.strftime("%Y-%m-%d_%H-%M")
directory = "{}_{}".format(timestamp, cfg.name)
if not os.path.exists(directory):
    os.makedirs(directory)


# Set up renderer
model = NeuralRenderer(
    img_shape=cfg.img_shape,
    channels_2d=cfg.channels_2d,
    strides_2d=cfg.strides_2d,
    channels_3d=cfg.channels_3d,
    strides_3d=cfg.strides_3d,
    num_channels_inv_projection=cfg.num_channels_inv_projection,
    num_channels_projection=cfg.num_channels_projection,
    mode=cfg.mode)

	
model.print_model_info()

model = model.to(device)


# set up trainer
trainer = Trainer(device, model, lr=cfg.lr, ssim_loss_weight=cfg.ssim_loss_weight)

dataloader = dataloader1(path_to_data = cfg.path_to_data,
                                     batch_size=cfg.batch_size,
                                     img_size=cfg.img_shape,
                                     crop_size=128)

# set up test loader
if cfg.path_to_test_data:
    test_dataloader = dataloader1(path_to_data = cfg.path_to_test_data,
                                              batch_size=cfg.batch_size,
                                              img_size=cfg.img_shape,
                                              crop_size=128)
else:
    test_dataloader = None

	
# Train renderer, save generated images, losses and model
trainer.train(dataloader, cfg.epochs, save_dir=directory,
              save_freq=cfg.save_freq, test_dataloader=test_dataloader)

print("Experiment name: {}".format(cfg.name))
print("Best training loss: {:.4f}".format(min(trainer.epoch_loss_history["total"])))
print("Best validation loss: {:.4f}".format(min(trainer.val_loss_history["total"])))
