import torch
import torch.nn as nn
from models.neural_renderer import get_swapped_indices
from torchvision.utils import save_image
import json
from pytorch_msssim import SSIM

	
	

class Trainer():
    def __init__(self, device, model, lr=2e-4, ssim_loss_weight=0.1):
        self.device = device
        self.model = model
        self.lr = lr
        self.ssim_loss_weight = ssim_loss_weight
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_func = nn.L1Loss()
        self.ssim_loss_func = SSIM(data_range=1.0, size_average=True,
                                       channel=3, nonnegative_ssim=False)

        # Loss log
        self.register_losses = True
        self.recorded_losses = ["total", "regress", "ssim"]
        self.loss_history = {loss_type: [] for loss_type in self.recorded_losses}
        self.epoch_loss_history = {loss_type: [] for loss_type in self.recorded_losses}
        self.val_loss_history = {loss_type: [] for loss_type in self.recorded_losses}

    def train(self, dataloader, epochs, save_dir=None, save_freq=1, test_dataloader=None):

        ### Loop all epochs!
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self.train_epoch(dataloader)

            for loss_type in self.recorded_losses:
                self.epoch_loss_history[loss_type].append(
                    sum(self.loss_history[loss_type][-len(dataloader):]) / len(dataloader)
                )
				
            # Print epoch losses
            print("Mean epoch loss:")
            self._print_losses()

            # Save generated results and model.
            if save_dir is not None:
                rendered = self.render_fixed_img()
                save_image(rendered.cpu(),
                           save_dir + "/imgs_gen_{}.png".format(str(epoch + 1).zfill(3)), nrow=6)
                # Save model
                if (epoch + 1) % save_freq == 0:
                    self.model.save(save_dir + "/model.pt")

            if test_dataloader is not None:
                regress_loss, ssim_loss, total_loss = test_loss(self, test_dataloader)
                print("Validation:\n regress: {:.4f}, ssim: {:.4f}, total: {:.4f}".format(regress_loss, ssim_loss, total_loss))
                self.val_loss_history["regress"].append(regress_loss)
                self.val_loss_history["ssim"].append(ssim_loss)
                self.val_loss_history["total"].append(total_loss)
            
                if min(self.val_loss_history["total"]) == total_loss:
                    print("Best model saved!!")
                    self.model.save(save_dir + "/best_model.pt")

        # Save final model
        if save_dir is not None:
            self.model.save(save_dir + "/model.pt")

    def train_epoch(self, dataloader):
        # Training for each epoch

        num_iterations = len(dataloader)
        for i, batch in enumerate(dataloader):

			      imgs, generated, scenes, scenes_rotated = self.model(batch)
			
			      self.optimizer.zero_grad()

			      loss_regress = self.loss_func(generated, imgs)           ############ Loss defined here!

        # maximize SSIM = minimize -SSIM
			      loss_ssim = 1. - self.ssim_loss_func(generated, imgs)
			      loss_total = loss_regress + self.ssim_loss_weight * loss_ssim

			      loss_total.backward()
			      self.optimizer.step()
			
			      if self.register_losses:
			         self.loss_history["total"].append(loss_total.item())
			         self.loss_history["regress"].append(loss_regress.item())
			         self.loss_history["ssim"].append(loss_ssim.item())
			   
            # Print losses
			      print("{}/{}".format(i + 1, num_iterations))
			      self._print_losses()


    def render_fixed_img(self):

        _, generated, _, _ = self.model(self.fixed_batch)
        return generated
		
    def _print_losses(self):
        """Prints most recent losses."""
        loss_info = []
        for loss_type in self.recorded_losses:   
            loss = self.loss_history[loss_type][-1]
            
            loss_info += [loss_type, loss]
        print("{}: {:.3f}, {}: {:.3f}, {}: {:.3f}".format(*loss_info))


def test_loss(trainer, dataloader):
    #compute mean loss of trained model for test data.

    with torch.no_grad():
        trainer.register_losses = False

        regress_loss = 0.
        ssim_loss = 0.
        total_loss = 0.
        for i, batch in enumerate(dataloader):
            imgs, generated, scenes, scenes_rotated = trainer.model(batch)

            # update loss
            current_regress_loss = trainer.loss_func(generated, imgs).item()

            current_ssim_loss = 1. - trainer.ssim_loss_func(generated, imgs).item()

            regress_loss += current_regress_loss
            ssim_loss += current_ssim_loss
            total_loss += current_regress_loss + trainer.ssim_loss_weight * current_ssim_loss

        # Average losses over dataset
        regress_loss /= len(dataloader)
        ssim_loss /= len(dataloader)
        total_loss /= len(dataloader)

        # continue training
        trainer.register_losses = True

    return regress_loss, ssim_loss, total_loss
