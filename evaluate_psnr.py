import sys
import torch
from misc.dataloaders import dataset_transform
from misc.evaluation import get_dataset_psnr
from models.neural_renderer import load_model

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compute_psnr")
    parser.add_argument("--model_path", default=None, type=str)  # image path
    parser.add_argument("--data_dir", default=None, type=str)  # model path
    args = parser.parse_args()

    model_path = args.model_path
	data_dir = args.data_dir


    # Load pretrained model
    model = load_model(model_path)
    model = model.to(device)

	# Load data
    dataset = dataset_transform(data_dir, img_size=(3, 128, 128),
                               crop_size=128, allow_odd_num_imgs=True)

    # Compute PSNR
    with torch.no_grad():
        psnrs = get_dataset_psnr(device, model, dataset, source_img_idx_shift=64,
                             batch_size=16)
