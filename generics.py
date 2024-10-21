from typing import TypeVar

from torch.nn import Conv2d, Conv3d, Conv1d,ConvTranspose2d, ConvTranspose3d,  LayerNorm


ConvType = TypeVar("ConvType", Conv2d, Conv3d, Conv1d, ConvTranspose2d, ConvTranspose3d)
NAType = TypeVar("NAType", bound="NAT")
ConvMultiHeadNAType = TypeVar("ConvMultiHeadNAType", bound="ConvMultiHeadNA")
ConvNATTransformerType = TypeVar("ConvNATTransformerType", bound="ConvNATTransformer")
TransformerStackType = TypeVar("TransformerStackType", bound="TransformerStack")
SharedConvNAType = TypeVar("SharedConvNAType", bound="SharedConvNA")