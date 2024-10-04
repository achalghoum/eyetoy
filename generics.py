from typing import TypeVar

from torch.nn import Conv2d, Conv3d, LayerNorm


ConvType = TypeVar("ConvType", Conv2d, Conv3d)
ConvNATType = TypeVar("ConvNATType", bound="ConvNAT")
ConvMultiHeadNATType = TypeVar("ConvMultiHeadNATType", bound="ConvMultiHeadNAT")
