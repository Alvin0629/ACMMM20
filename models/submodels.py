import torch
import torch.nn as nn
from models.layers import ResBlock2d, ResBlock3d, num_channels_to_num_groups


class ResNet2d(nn.Module):
    #ResNets taking 2d inputs.

    def __init__(self, input_shape, channels, strides, final_conv_channels=0,
                 filter_multipliers=(1, 1), add_groupnorm=True):
        super(ResNet2d, self).__init__()
        assert len(channels) == len(strides)
        self.input_shape = input_shape
        self.channels = channels
        self.strides = strides
        self.filter_multipliers = filter_multipliers
        self.add_groupnorm = add_groupnorm

 
        output_channels, output_height, output_width = input_shape

        for stride in strides:
            if stride == 1:
                pass
            elif stride == 2:
                output_height //= 2
                output_width //= 2
            elif stride == -2:
                output_height *= 2
                output_width *= 2

        self.output_shape = (channels[-1], output_height, output_width)


        forward_layers = [
            nn.Conv2d(self.input_shape[0], channels[0], kernel_size=1,
                      stride=1, padding=0)
        ]
        in_channels = channels[0]
        multiplier1x1, multiplier3x3 = filter_multipliers
        for out_channels, stride in zip(channels, strides):
            if stride == 1:
                forward_layers.append(
                    ResBlock2d(in_channels,
                              [out_channels * multiplier1x1, out_channels * multiplier3x3],
                               add_groupnorm=add_groupnorm)
                )
            if stride == 2:
                forward_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1)
                )
            if stride == -2:
                forward_layers.append(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1)
                )

            # Add non-linearity activation function
            if stride == 2 or stride == -2:
                forward_layers.append(nn.GroupNorm(num_channels_to_num_groups(out_channels), out_channels))
                forward_layers.append(nn.LeakyReLU(0.2, True))

            in_channels = out_channels

        if final_conv_channels:
            forward_layers.append(
                nn.Conv2d(in_channels, final_conv_channels, kernel_size=1,
                          stride=1, padding=0)
            )

        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, inputs):
        return self.forward_layers(inputs)


class ResNet3d(nn.Module):
    #ResNets taking 3d inputs.

    
    def __init__(self, input_shape, channels, strides, final_conv_channels=0,
                 filter_multipliers=(1, 1), add_groupnorm=True):
        super(ResNet3d, self).__init__()
        assert len(channels) ==  len(strides)
        self.input_shape = input_shape
        self.channels = channels
        self.strides = strides
        self.filter_multipliers = filter_multipliers
        self.add_groupnorm = add_groupnorm


        output_channels, output_depth, output_height, output_width = input_shape

        for stride in strides:
            if stride == 1:
                pass
            elif stride == 2:
                output_depth //= 2
                output_height //= 2
                output_width //= 2
            elif stride == -2:
                output_depth *= 2
                output_height *= 2
                output_width *= 2

        self.output_shape = (channels[-1], output_depth, output_height, output_width)

        forward_layers = [
            nn.Conv3d(self.input_shape[0], channels[0], kernel_size=1,
                      stride=1, padding=0)
        ]
        in_channels = channels[0]
        multiplier1x1, multiplier3x3 = filter_multipliers
        for out_channels, stride in zip(channels, strides):
            if stride == 1:
                forward_layers.append(
                    ResBlock3d(in_channels,
                              [out_channels * multiplier1x1, out_channels * multiplier3x3],
                               add_groupnorm=add_groupnorm)
                )
            if stride == 2:
                forward_layers.append(
                    nn.Conv3d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1)
                )
            if stride == -2:
                forward_layers.append(
                    nn.ConvTranspose3d(in_channels, out_channels, kernel_size=4,
                              stride=2, padding=1)
                )


            if stride == 2 or stride == -2:
                forward_layers.append(nn.GroupNorm(num_channels_to_num_groups(out_channels), out_channels))
                forward_layers.append(nn.LeakyReLU(0.2, True))

            in_channels = out_channels

        if final_conv_channels:
            forward_layers.append(
                nn.Conv3d(in_channels, final_conv_channels, kernel_size=1,
                          stride=1, padding=0)
            )

        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, inputs):
        return self.forward_layers(inputs)


class Projection(nn.Module):
    # 3D voxel-like features to a 2D image-like feature vector

    def __init__(self, input_shape, num_channels):
        super(Projection, self).__init__()
        self.input_shape = input_shape
        self.num_channels = num_channels
        self.output_shape = (num_channels[-1],) + input_shape[2:]

        in_channels = self.input_shape[0] * self.input_shape[1]
        forward_layers = []
        num_layers = len(num_channels)
        for i in range(num_layers):
            out_channels = num_channels[i]
            forward_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
            # Add LeakyReLU, except for last layer
            if i != num_layers - 1:
                forward_layers.append(nn.GroupNorm(num_channels_to_num_groups(out_channels), out_channels))
                forward_layers.append(nn.LeakyReLU(0.2, True))
            in_channels = out_channels
 
        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, inputs):
        #Reshapes from 3D -> 2D

        batch_size, channels, depth, height, width = inputs.shape
        reshaped = inputs.view(batch_size, channels * depth, height, width)
        # 1x1 conv
        return self.forward_layers(reshaped)


class Inv_Projection(nn.Module):
    #inverse projection from a 2D feature to 3D feature

    def __init__(self, input_shape, num_channels):
        super(Inv_Projection, self).__init__()
        self.input_shape = input_shape
        self.num_channels = num_channels
        assert num_channels[-1] % input_shape[-1] == 0
        self.output_shape = (num_channels[-1] // input_shape[-1], input_shape[-1]) + input_shape[1:]

        in_channels = self.input_shape[0]
        forward_layers = []
        num_layers = len(num_channels)
        for i in range(num_layers):
            out_channels = num_channels[i]
            forward_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,
                          padding=0)
            )
            # Add LeakyReLU, except for last layer
            if i != num_layers - 1:
                forward_layers.append(nn.GroupNorm(num_channels_to_num_groups(out_channels), out_channels))
                forward_layers.append(nn.LeakyReLU(0.2, True))
            in_channels = out_channels

        self.forward_layers = nn.Sequential(*forward_layers)

    def forward(self, inputs):
        # 1x1 conv
        features = self.forward_layers(inputs)
        batch_size = inputs.shape[0]
        return features.view(batch_size, *self.output_shape)
