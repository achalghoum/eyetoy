from typing import TypeVar

from torch.nn import Conv2d, Conv3d, Conv1d,ConvTranspose2d, ConvTranspose3d,  LayerNorm


ConvType = TypeVar("ConvType", Conv2d, Conv3d, Conv1d, ConvTranspose2d, ConvTranspose3d)
NAType = TypeVar("NAType", bound="NAT")
MultiScaleMultiHeadNAType = TypeVar("MultiScaleMultiHeadNAType", bound="MulitScaleMultiHeadNA")
MSNATTransformerType = TypeVar("MSNATTransformerType", bound="MSNATTransformer")
TransformerStackType = TypeVar("TransformerStackType", bound="TransformerStack")
SharedScaleNAType = TypeVar("SharedScaleNAType", bound="SharedScaleNA")