import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_filter_count(layer_idx, filters_root):
    return 2 ** layer_idx * filters_root


class Pad2D(nn.Module):
    """ Padding for 2D input (e.g. images).

    Args:
        padding: tuple of 2 ints, how many zeros to add at the beginning and at the end of
            the 2 padding dimensions (rows and cols)
        mode: "constant", "reflect", or "replicate"
    """
    def __init__(self, padding=(1, 1), mode="constant"):
        super(Pad2D, self).__init__()
        if mode not in ["constant", "reflect", "replicate", "valid"]:
            raise ValueError("Padding mode must be 'valid', 'constant', 'reflect', or 'replicate'")
        self.padding = (padding[1], padding[1], padding[0], padding[0])  # PyTorch expects padding in reverse order
        self.mode = mode

    def forward(self, x):
        """Pad 2D input tensor x.
        
        Args:
            x (torch.Tensor): 4D input tensor (B, C, H, W)
        Returns:
            torch.Tensor: 4D output tensor (B, C, H + 2*padding[0], W + 2*padding[1])
        """
        if self.mode =="valid":
            return x
        else:
            return nn.functional.pad(x, self.padding, mode=self.mode)


def maybe_crop(x, target_shape, data_format="channels_first"):
    """Center crops x to target_shape if necessary.

    Args:
        x (torch.Tensor): input tensor
        target_shape (list): shape of a reference tensor in BHWC or BCHW format
        data_format (str): data format, either "channels_last" or "channels_first"
    Returns:
        x (torch.Tensor): cropped tensor
    """
    if data_format == "channels_last":
        target_shape = target_shape[1:3]
        x_shape = x.size()[1:3]
        h_diff = x_shape[0] - target_shape[0]
        w_diff = x_shape[1] - target_shape[1]
        h_crop_start = h_diff // 2
        w_crop_start = w_diff // 2
        h_crop_end = h_diff - h_crop_start
        w_crop_end = w_diff - w_crop_start
        x = x[:, h_crop_start:-h_crop_end, w_crop_start:-w_crop_end, :]
    elif data_format == "channels_first":
        target_shape = target_shape[2:4]
        x_shape = x.size()[2:4]
        h_diff = x_shape[0] - target_shape[0]
        w_diff = x_shape[1] - target_shape[1]
        h_crop_start = h_diff // 2
        w_crop_start = w_diff // 2
        h_crop_end = h_diff - h_crop_start
        w_crop_end = w_diff - w_crop_start
        x = x[:, :, h_crop_start:-h_crop_end, w_crop_start:-w_crop_end]
    return x


class ConvBlock(nn.Module):
    """Convolutional block consisting of two convolutional layers with same number of filters
        and a batch normalization layer in between.

    Args:
        layer_idx (int): index of the layer, used to compute the number of filters
        filters_root (int): number of filters in the first convolutional layer
        kernel_size (int): size of convolutional kernels
        padding: padding, either "VALID", "CONSTANT", "REFLECT", or "SYMMETRIC"
            activation: activation to be used
        data_format: data format, either "channels_last" or "channels_first"
    """
    def __init__(
            self, layer_idx, filters_root, kernel_size, padding, activation, up=False, **kwargs,
        ):
        super(ConvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.padding=padding
        if layer_idx == 0:
            # get in_channels from kwargs
            in_ch = kwargs.get('in_channels', 2)
        self.activation=getattr(nn, activation)()
        filters = _get_filter_count(layer_idx, filters_root)
        self.padding_layer = Pad2D(padding=(1, 1), mode=self.padding)

        if up:
            self.conv2d_0 = nn.Conv2d(
                in_channels=_get_filter_count(layer_idx+1, filters_root),
                out_channels=filters, kernel_size=(1, 1), stride=1, padding='valid'
            )
        else:
            self.conv2d_0 = nn.Conv2d(
                in_channels=_get_filter_count(layer_idx-1, filters_root) if layer_idx != 0 else in_ch,
                out_channels=filters, kernel_size=(1, 1), stride=1, padding='valid'
            )
        self.conv2d_1 = nn.Conv2d(
            in_channels=filters, out_channels=filters, 
            kernel_size=(3, 3), stride=1, padding='valid'
        )
        self.bn_1 = nn.BatchNorm2d(filters)
        self.conv2d_2 = nn.Conv2d(
            in_channels=filters, out_channels=filters,kernel_size=(3, 3), stride=1, padding='valid'
        )
        self.bn_2 = nn.BatchNorm2d(filters)

    def forward(self, x):
        """Apply ConvBlock to inputs.

        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor
        """
        skip = self.conv2d_0(x)
        x = self.padding_layer(skip)
        x = self.conv2d_1(x)
        x = self.bn_1(x)
        x = self.activation(x)
        x = self.padding_layer(x)
        x = self.conv2d_2(x)
        x = self.bn_2(x)
        x = self.activation(x)
        if self.padding == "valid":
            skip = maybe_crop(skip, x.size())
        x = x + skip
        return x


class UpconvBlock(nn.Module):
    """Upconvolutional block consisting of an upsampling layer and a convolutional layer.

    Args:
        layer_idx (int): index of the layer, used to compute the number of filters
        filters_root (int): number of filters in the first convolutional layer
        kernel_size (tuple): size of convolutional kernels
        pool_size (tuple): size of the pooling layer
        padding (str): padding, either "VALID", "CONSTANT", "REFLECT", or "SYMMETRIC"
        activation (fn): activation to be used
        data_format (str): data format, either "channels_last" or "channels_first"
    """
    def __init__(
            self, layer_idx, filters_root, kernel_size, pool_size, padding, activation, **kwargs
        ):
        super(UpconvBlock, self).__init__(**kwargs)
        self.layer_idx=layer_idx
        self.filters_root=filters_root
        self.kernel_size=kernel_size
        self.pool_size=pool_size
        self.padding=padding
        self.activation=activation

        filters = _get_filter_count(layer_idx, filters_root)
        self.padding_layer = Pad2D(padding=(1, 1), mode=self.padding)
        self.upconv = nn.ConvTranspose2d(
            in_channels=filters, out_channels=filters // 2,
            kernel_size=pool_size, stride=pool_size, padding=0
        )
        self.activation_1 = getattr(nn, activation)()

    def forward(self, x):
        """Apply UpconvBlock to inputs.
        
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor
        """
        # x = self.padding_layer(x)
        x = self.upconv(x)
        x = self.activation_1(x)
        return x


class CropConcatBlock(nn.Module):
    """CropConcatBlock that crops spatial dimensions and concatenates filter maps.

    Args:
        data_format (str): data format, either "channels_last" or "channels_first"
    """
    def __init__(self, **kwargs):
        super(CropConcatBlock, self).__init__(**kwargs)

    def forward(self, x, down_layer):
        """Apply CropConcatBlock to inputs.

        Args:
            x (torch.Tensor): input tensor
            down_layer (torch.Tensor): tensor from the contracting path
        Returns:
            torch.Tensor: output tensor
        """
        x1_shape = down_layer.shape
        x2_shape = x.shape
        height_diff = abs(x1_shape[2] - x2_shape[2]) // 2
        width_diff = abs(x1_shape[3] - x2_shape[3]) // 2
        down_layer_cropped = down_layer[:,:,
                                        height_diff: (x2_shape[2] + height_diff),
                                        width_diff: (x2_shape[3] + width_diff)]
        x = torch.cat([down_layer_cropped, x], dim=1)
        return x


class UNet(nn.Module):
    """UNet model with contracting and expanding paths.

    Args:
        num_classes (int): number of classes
        layer_depth (int): number of layers
        filters_root (int): number of filters in the first convolutional layer
        data_format: data format, either "channels_last" or "channels_first"
        kernel_size (int): size of convolutional kernels
        pool_size (int): size of the pooling layer
        padding (str): padding, either "VALID", "CONSTANT", "REFLECT", or "SYMMETRIC"
        activation (str): activation to be used
    """
    def __init__(self, 
                 num_classes: int = 2,
                 layer_depth: int = 5,
                 filters_root: int = 64,
                 data_format = "channels_first",
                 kernel_size: int = 3,
                 pool_size: int = 2,
                 padding: str = "reflect",
                 activation: str = 'ReLU'):
        super(UNet, self).__init__()

        self.layer_depth = layer_depth
        self.contracting_layers = nn.ModuleList()
        self.expanding_layers = nn.ModuleList()
        self.padding = padding

        for layer_idx in range(layer_depth-1):
            conv_block = ConvBlock(
                layer_idx = layer_idx, filters_root=filters_root, kernel_size=kernel_size,
                padding=self.padding, activation=activation)
            self.contracting_layers.append(conv_block)
            self.contracting_layers.append(
                nn.MaxPool2d(kernel_size=(pool_size, pool_size))
            )
        self.bottle_neck = ConvBlock(
        layer_idx=layer_idx+1, filters_root=filters_root, kernel_size=kernel_size,
        padding=self.padding, activation=activation
        )
        for layer_idx in range(layer_depth-2, -1, -1):
            upconv_block = UpconvBlock(
                layer_idx=layer_idx+1, filters_root=filters_root, kernel_size=kernel_size,
                padding=self.padding, activation=activation, pool_size=(pool_size, pool_size)
            )
            crop_concat_block = CropConcatBlock()
            conv_block = ConvBlock(
            layer_idx=layer_idx, filters_root=filters_root, kernel_size=kernel_size,
            padding=self.padding, activation=activation, up=True
            )
            self.expanding_layers += [upconv_block, crop_concat_block, conv_block]

        self.final_conv = nn.Conv2d(filters_root, num_classes, kernel_size=(1, 1))
        if data_format == "channels_last":
            self.to(memory_format=torch.channels_last)

    def forward(self, x):
        """Forward pass of the UNet model.
        
        Args:
            x (torch.Tensor): input tensor
        Returns:
            torch.Tensor: output tensor
        """
        contracting_outputs = []
        for layer in self.contracting_layers:
            x = layer(x)
            if isinstance(layer, ConvBlock):
                contracting_outputs.append(x)
        x = self.bottle_neck(x)

        for layer in self.expanding_layers:
            if isinstance(layer, CropConcatBlock):
                x = layer(x, contracting_outputs.pop(-1))
            else:
                x = layer(x)
        x = self.final_conv(x)
        return torch.sigmoid(x)
