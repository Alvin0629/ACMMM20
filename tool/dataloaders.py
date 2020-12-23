import glob
import json
import torch
from numpy import float32 as np_float32
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms


def dataloader1(path_to_data='chairs', batch_size=32,
                            img_size=(3, 128, 128), crop_size=128):

    assert batch_size % 2 == 0

    dataset = dataset_transform(path_to_data, img_size, crop_size)

    sampler = Randomsample(dataset)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)


def dataset_transform(path_to_data='chairs', img_size=(3, 128, 128),
                         crop_size=128):

    img_transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(img_size[1:]),
        transforms.ToTensor()
    ])

    dataset = SceneRenderDataset(path_to_data=path_to_data,
                                 img_transform=img_transform)

    return dataset


class SceneRenderDataset(Dataset):
   # Load multi-view images and the corresponding camera angles.   
        #Image paths must be of the form "00002.png"         
        #Angles will be in degrees.
  
    def __init__(self, path_to_data='chairs', img_transform=None):
        self.path_to_data = path_to_data
        self.img_transform = img_transform
        self.data = []
        # each folder contains a single scene with different parameters and views
        self.scene_paths = glob.glob(path_to_data + '/*')
        self.scene_paths.sort()  # ensure consistent ordering
        self.num_scenes = len(self.scene_paths)
        self.num_imgs_per_scene = len(glob.glob(self.scene_paths[0] + '/*.png'))
        # If not even, drop last one
        if self.num_imgs_per_scene % 2 != 0:
            self.num_imgs_per_scene -= 1
        # For each scene, extract image and angle parameters
        for scene_path in self.scene_paths:
            scene_name = scene_path.split('/')[-1]
			
            with open(scene_path + '/render_params.json') as f:
                render_params = json.load(f)

            img_paths = glob.glob(scene_path + '/*.png')
            img_paths.sort()  # ensure consistent ordering
            img_paths = img_paths[:self.num_imgs_per_scene]

            for img_path in img_paths:
                img_file = img_path.split('/')[-1]
                img_idx = img_file.split('.')[0][-5:]
                # parameters to float32
                img_params = {key: np_float32(value)
                              for key, value in render_params[img_idx].items()}
                self.data.append({
                    "scene_name": scene_name,
                    "img_path": img_path,
                    "img_params": img_params
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]["img_path"]
        img_params = self.data[idx]["img_params"]

        img = Image.open(img_path)

        if self.img_transform:
            img = self.img_transform(img)

        collected_data = {
            "img": img[0:3], ##deal with 4-channel png img.
            "scene_name": self.data[idx]["scene_name"],
            "img_params": self.data[idx]["img_params"]
        }

        return collected_data


class Randomsample(Sampler):
  
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        num_scenes = self.dataset.num_scenes
        num_imgs_per_scene = self.dataset.num_imgs_per_scene

        # Sample scene. number of image in a scene // 2
        scene_sample = [torch.randperm(num_scenes) for _ in range(num_imgs_per_scene // 2)]
        # Sample image per scene
        img_samples = [torch.randperm(num_imgs_per_scene) for _ in range(num_scenes)]

        final_data_sample = []

        for i, scene_sampled in enumerate(scene_sample):
            for scene_idx in scene_sampled:
                img_sample = img_samples[scene_idx]
                # append two images each time.
                final_data_sample.append(scene_idx.item() * num_imgs_per_scene + img_sample[2*i].item())
                final_data_sample.append(scene_idx.item() * num_imgs_per_scene + img_sample[2*i + 1].item())

        return iter(final_data_sample)

    def __len__(self):
        return len(self.dataset)

