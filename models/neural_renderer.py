import torch
import torch.nn as nn
from tool.utils import pretty_print_layers_info, count_parameters
from models.submodels import ResNet2d, ResNet3d, Projection, Inv_Projection
from models.rotation_layers import SphericalMask, Rotate3d


class NeuralRenderer(nn.Module):
    ##render: 3d->2d; scene view->resnet3d->projection->resnet2d->2d image
	  ##inv_render: 2d->3d: 2d image->resnet2d->inv_projection->resnet3d->scene view


    def __init__(self, img_shape, channels_2d, strides_2d, channels_3d,
                 strides_3d, num_channels_inv_projection, num_channels_projection,
                 mode='bilinear'):
        super(NeuralRenderer, self).__init__()
        self.img_shape = img_shape
        self.channels_2d = channels_2d
        self.strides_2d = strides_2d
        self.channels_3d = channels_3d
        self.strides_3d = strides_3d
        self.num_channels_projection = num_channels_projection
        self.num_channels_inv_projection = num_channels_inv_projection
        self.mode = mode


        self.inv_transform_2d = ResNet2d(self.img_shape, channels_2d,
                                         strides_2d)

        input_shape = self.inv_transform_2d.output_shape
        self.inv_projection = Inv_Projection(input_shape, num_channels_inv_projection)


        self.inv_transform_3d = ResNet3d(self.inv_projection.output_shape,
                                         channels_3d, strides_3d)

        self.rotation_layer = Rotate3d(self.mode)


        forward_channels_3d = list(reversed(channels_3d))[1:] + [channels_3d[0]]
        forward_strides_3d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_3d[1:]))] + [strides_3d[0]]
        self.transform_3d = ResNet3d(self.inv_transform_3d.output_shape,
                                     forward_channels_3d, forward_strides_3d)

									 
        self.projection = Projection(self.transform_3d.output_shape,
                                     num_channels_projection)


        forward_channels_2d = list(reversed(channels_2d))[1:] + [channels_2d[0]]
        forward_strides_2d = [-stride if abs(stride) == 2 else 1 for stride in list(reversed(strides_2d[1:]))] + [strides_2d[0]]
        final_conv_channels_2d = img_shape[0]
        self.transform_2d = ResNet2d(self.projection.output_shape,
                                     forward_channels_2d, forward_strides_2d,
                                     final_conv_channels_2d)

        self.scene_shape = self.inv_transform_3d.output_shape
        self.spherical_mask = SphericalMask(self.scene_shape)

    def render(self, scene):
   
        #input scene view: (batch_size, channels, depth, height, width).
 
        features_3d = self.transform_3d(scene)
        features_2d = self.projection(features_3d)
        return torch.sigmoid(self.transform_2d(features_2d))

    def inverse_render(self, img):
       
        #input img: (batch_size, channels, height, width).
      
        features_2d = self.inv_transform_2d(img)
        features_3d = self.inv_projection(features_2d)
        scene = self.inv_transform_3d(features_3d)
        return self.spherical_mask(scene)

    def rotate(self, scene, rotation_matrix):

        return self.rotation_layer(scene, rotation_matrix)

    def rotate_source_to_target(self, scene, azimuth_source, elevation_source,
                                azimuth_target, elevation_target):
   
        return self.rotation_layer.rotate_source_to_target(scene,
                                                           azimuth_source,
                                                           elevation_source,
                                                           azimuth_target,
                                                           elevation_target)

    def forward(self, batch):
       
        device = self.spherical_mask.mask.device
        imgs = batch["img"].to(device)
        params = batch["img_params"]
        azimuth = params["azimuth"].to(device)
        elevation = params["elevation"].to(device)

        scenes = self.inverse_render(imgs)

        swapped_idx = get_swapped_indices(azimuth.shape[0])

        azimuth_swapped = azimuth[swapped_idx]
        elevation_swapped = elevation[swapped_idx]
        scenes_swapped = \
            self.rotate_source_to_target(scenes, azimuth, elevation,
                                         azimuth_swapped, elevation_swapped)


        scenes_rotated = scenes_swapped[swapped_idx]

        rendered = self.render(scenes_rotated)

        return imgs, rendered, scenes, scenes_rotated

    def print_model_info(self):

        print("Forward renderer")
        print("----------------\n")
        pretty_print_layers_info(self.transform_3d, "3D Layers")
        print("\n")
        pretty_print_layers_info(self.projection, "Projection")
        print("\n")
        pretty_print_layers_info(self.transform_2d, "2D Layers")
        print("\n")

        print("Inverse renderer")
        print("----------------\n")
        pretty_print_layers_info(self.inv_transform_2d, "Inverse 2D Layers")
        print("\n")
        pretty_print_layers_info(self.inv_projection, "Inverse Projection")
        print("\n")
        pretty_print_layers_info(self.inv_transform_3d, "Inverse 3D Layers")
        print("\n")

        print("Scene Representation:")
        print("\tShape: {}".format(self.scene_shape))
        print("\tSize: {}\n".format(int(self.spherical_mask.mask.sum().item())))

        print("Number of parameters: {}\n".format(count_parameters(self)))

    def get_model_config(self):
        """Returns the complete model configuration as a dict."""
        return {
            "img_shape": self.img_shape,
            "channels_2d": self.channels_2d,
            "strides_2d": self.strides_2d,
            "channels_3d": self.channels_3d,
            "strides_3d": self.strides_3d,
            "num_channels_inv_projection": self.num_channels_inv_projection,
            "num_channels_projection": self.num_channels_projection,
            "mode": self.mode
        }

    def save(self, filename):

        torch.save({
            "config": self.get_model_config(),
            "state_dict": self.state_dict()
        }, filename)


def load_model(filename):

    model_dict = torch.load(filename, map_location="cpu")
    config = model_dict["config"]

    model = NeuralRenderer(
        img_shape=config["img_shape"],
        channels_2d=config["channels_2d"],
        strides_2d=config["strides_2d"],
        channels_3d=config["channels_3d"],
        strides_3d=config["strides_3d"],
        num_channels_inv_projection=config["num_channels_inv_projection"],
        num_channels_projection=config["num_channels_projection"],
        mode=config["mode"]
    )

    model.load_state_dict(model_dict["state_dict"])
    return model


def get_swapped_indices(length):
    return [i + 1 if i % 2 == 0 else i - 1 for i in range(length)]
