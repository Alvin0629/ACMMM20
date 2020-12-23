import matplotlib.pyplot as plt
import torch
import torchvision
import imageio
from torchvision.transforms import ToTensor
from tool.visualize import generate_novel_views


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_img_tensor(img, nrow=6):  ## img (BS, C, H, W)

    img_grid = torchvision.utils.make_grid(img, nrow=nrow)
    plt.imshow(img_grid.cpu().numpy().transpose(1, 2, 0))
	plt.show()
	
	
		
from models.neural_renderer import load_model

# Load pretrained model
model = load_model('chair.pt').to(device)



# load test image.
img = imageio.imread('test_example/chair.png')[:3]
plt.imshow(img)
plt.show()


## Now Infer !!!!
test_img = ToTensor()(img)
test_img = test_img.unsqueeze(0).to(device)


# Infer 3D view
scene = model.inverse_render(test_img)


# Generate 2D image.
result = model.render(scene)



# Input source camera azimuth and elevation in degree
azimuth_source = torch.Tensor([45.]).to(device)
elevation_source = torch.Tensor([30.]).to(device)



# Input target camera azimuth and elevation in degree for view generation.
azimuth_shifts = torch.Tensor([20., -50., 120., 180., -90., 50.]).to(device)
elevation_shifts = torch.Tensor([10., -30., 40., -70., 10., 30.]).to(device)


views = generate_novel_views(model, img_source[0], azimuth_source, elevation_source,
                             azimuth_shifts, elevation_shifts)

plot_img_tensor(views.detach(), nrow=2)

