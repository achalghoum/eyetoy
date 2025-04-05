from typing import TypeVar

from torch.nn import Conv2d, Conv3d, Conv1d, ConvTranspose2d, ConvTranspose3d, LayerNorm


ConvType = TypeVar("ConvType", Conv2d, Conv3d, Conv1d, ConvTranspose2d, ConvTranspose3d)
NAType = TypeVar("NAType", bound="NAT") # type: ignore
MultiScaleMultiHeadNAType = TypeVar(
    "MultiScaleMultiHeadNAType", bound="MulitScaleMultiHeadNA" # type: ignore
)
MSNATTransformerType = TypeVar("MSNATTransformerType", bound="MSNATTransformer") # type: ignore
TransformerStackType = TypeVar("TransformerStackType", bound="TransformerStack") # type: ignore
SharedScaleNAType = TypeVar("SharedScaleNAType", bound="SharedScaleNA") # type: ignore
