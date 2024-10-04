from dataclasses import dataclass
from typing import List, Optional


class ConvParams(dataclass):
    kernel_size: int
    activation: Optional[str] = "swiglu"
    dilation: Optional[int] = 0
    in_channels: Optional[int]
    out_channels: Optional[int]
    stride: Optional[int] = 1
    padding: Optional[int]
    bias: Optional[int]


class NeighborhoodAttentionParams(dataclass):
    min_kernel_size: Optional[int] = 5
    max_kernel_size: Optional[int] = 17
    min_dilation: Optional[int] = 0
    max_dilation: Optional[int] = 3


class HeadParams(dataclass):
    in_channels: Optional[int]
    intermediate_channels: Optional[int]
    out_channels: Optional[int]
    shared_conv_params: ConvParams
    attn_params: Optional[NeighborhoodAttentionParams] = NeighborhoodAttentionParams()
    is_causal: Optional[bool] = False


class MultiHeadAttentionParams(dataclass):
    num_heads: int
    head_params: List[HeadParams]
    in_channels: int
    intermediate_channels: Optional[int] = None
    out_channels: Optional[int] = None
    final_conv_params: ConvParams
    stride: Optional[int] = 1
    padding: Optional[int] = 0

    def __post_init__(self):
        if len(self.head_params) != self.num_heads:
            raise ValueError("Number of heads must match number of head params")
        if self.intermediate_channels is None:
            self.intermediate_channels = self.in_channels // self.num_heads
        if self.out_channels is None:
            self.out_channels = self.in_channels
        for head in self.head_params:
            if head.intermediate_channels is None:
                head.intermediate_channels = self.intermediate_channels
            if head.out_channels is None:
                head.out_channels = self.intermediate_channels
            if head.in_channels is None:
                head.in_channels = self.in_channels
        if self.final_conv_params is None:
            self.final_conv_params = ConvParams(kernel_size=1, in_channels=sum(
                head.out_channels for head in self.head_params), out_channels=self.out_channels)


DEFAULT_NUM_HEADS = 8
DEFAULT_HEAD_PARAMS = HeadParams()
COMPRESSION_HEAD_PARAMS = HeadParams()


class TransformerBlockParams(dataclass):
    attention_params: Optional[MultiHeadAttentionParams]
    final_conv_params: Optional[ConvParams]
    use_gated_residuals: bool = True
    gated_residual_conv_params: Optional[ConvParams] = None
    in_channels: int
    out_channels: int

    def __post_init__(self):
        if not self.attention_params:
            self.attention_params = MultiHeadAttentionParams(num_heads=DEFAULT_NUM_HEADS,
                                                             head_params=[
                                                                 DEFAULT_NUM_HEADS * DEFAULT_HEAD_PARAMS],
                                                             in_channels=self.in_channels,
                                                             out_channels=self.out_channels)
        if not self.final_conv_params:
            self.final_conv_params = ConvParams(kernel_size=1,
                                                in_channels=self.attention_params.out_channels,
                                                out_channels=self.out_channels)
        if self.use_gated_residuals and self.gated_residual_conv_params is None:
            self.gated_residual_conv_params = ConvParams(kernel_size=1,
                                                         in_channels=self.attention_params.out_channels + self.final_conv_params.out_channels,
                                                         out_channels=self.out_channels,
                                                         activation="sigmoid")
