from typing import TypeVar

from torch.nn import Conv2d, Conv3d, Conv1d,ConvTranspose2d, ConvTranspose3d,  LayerNorm


ConvType = TypeVar("ConvType", Conv2d, Conv3d, Conv1d, ConvTranspose2d, ConvTranspose3d)
ConvNATType = TypeVar("ConvNATType", bound="ConvNAT")
ConvMultiHeadNATType = TypeVar("ConvMultiHeadNATType", bound="ConvMultiHeadNAT")
ConvNATTransformerType = TypeVar("ConvNATTransformerType", bound="ConvNATTransformer")
TransformerStackType = TypeVar("TransformerStackType", bound="TransformerStack")
