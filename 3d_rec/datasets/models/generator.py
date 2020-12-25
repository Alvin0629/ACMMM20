from os import makedirs
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from networks.networks import ImageEncoder, VoxelDecoder
from basemodel import BaseModel
sys.path.append("..") 
from util import util_camera.project_3DPoints as project_3DPoints

class Model(BaseModel):

    def __init__(self, opt, logger):
        super(Model, self).__init__(opt, logger)
        self.voxel_key = 'voxel'
        self.requires = ['rgb', 'silhou', 'voxel']
        self.net = Net(2)
        self.criterion = nn.BCEWithLogitsLoss(reduction='elementwise_mean')
        self.optimizer = self.adam(
            self.net.parameters(),
            lr=opt.lr,
            **self.optim_params
        )
        self._nets = [self.net]
        self._optimizers.append(self.optimizer)
        self.input_names = ['rgb', 'silhou']
        self.gt_names = [voxel_key]
        self.init_vars(add_path=True)
        self._metrics = ['loss']
        self.init_weight(self.net)

    def train_on_batch(self, epoch, batch_idx, batch):
        self.net.zero_grad()
        pred = self.predict(batch)
        loss, loss_data = self.compute_loss(pred, sup=False)
        loss.backward()
        self.optimizer.step()
        batch_size = len(batch['rgb_path'])
        batch_log = {'size': batch_size, **loss_data}
        return batch_log

    def valid_on_batch(self, epoch, batch_idx, batch):
        pred = self.predict(batch, no_grad=True)
        _, loss_data = self.compute_loss(pred, sup=False)
        if np.mod(epoch, self.opt.vis_every_valid) == 0:
            if batch_idx < self.opt.vis_batches_valid:
                outdir = join(self.full_logdir, 'epoch%04d_valid' % epoch)
                makedirs(outdir, exist_ok=True)
                output = self.pack_output(pred, batch)
                self.visualizer.visualize(output, batch_idx, outdir)
                np.savez(join(outdir, 'batch%04d' % batch_idx), **output)
        batch_size = len(batch['rgb_path'])
        batch_log = {'size': batch_size, **loss_data}
        return batch_log

    def pack_output(self, pred, batch, add_gt=True):
        out = {}
        out['rgb_path'] = batch['rgb_path']
        out['pred_voxel'] = pred.detach().cpu().numpy()
        if add_gt:
            out['gt_voxel'] = batch[self.voxel_key].numpy()
            out['silhou_path'] = batch['silhou_path']
        return out

    def compute_loss(self, pred, sup = False):
	    if sup = True:
           loss = self.criterion(pred, getattr(self._gt, self.voxel_key))
		else:
		   projection_silhou = project_3DPoints(pred, H) 
		   is_mask = projection_silhou > 0
		   projection_silhou[is_mask] = 1
		   loss = self.criterion(projection_silhou, out['silhou_path'])
        loss_data = {}
        loss_data['loss'] = loss.mean().item()
        return loss, loss_data


class Net(nn.Module):

    def __init__(self, in_planes, encode_dims=200, silhou_thres=0):
        super().__init__()
        self.encoder = ImageEncoder(in_planes, encode_dims=encode_dims)
        self.decoder = VoxelDecoder(n_dims=encode_dims, nf=512)
        self.silhou_thres = silhou_thres

    def forward(self, input_struct):
        rgb = input_struct.rgb
        silhou = input_struct.silhou
        # Mask
        is_bg = silhou <= self.silhou_thres
        silhou[is_bg] = 0
        x = torch.cat((silhou, rgb), 1) 
        # Forward
        latent_vec = self.encoder(x)
        vox = self.decoder(latent_vec)
        return vox
