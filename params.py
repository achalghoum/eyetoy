from copy import deepcopy
from dataclasses import dataclass
from math import gcd
from typing import Callable, List, Optional, Union


@dataclass
class ConvParams:
    kernel_size: int
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    padding: Union[str,int] = "valid"
    bias: bool = True
    dilation: Optional[int] = 1
    stride: int = 1
    groups: int = 32

    def __post_init__(self):
        self._update_groups()

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ['in_channels', 'out_channels', 'groups']:
            self._update_groups()

    def _update_groups(self):
        if self.in_channels is not None and self.out_channels is not None:
            if self.in_channels % self.groups != 0 or self.out_channels % self.groups != 0:
                self.groups = gcd(self.in_channels, self.out_channels)


CONV_1X1 = ConvParams(kernel_size=1, stride=1, padding="same", groups=1)
CONV_3X3 = ConvParams(kernel_size=3, stride=3)
CONV_5X5 = ConvParams(kernel_size=5, stride=5)
CONV_7X7 = ConvParams(kernel_size=7, stride=7)
CONV_9X9 = ConvParams(kernel_size=9, stride=9)

CONV_3X3_SAME = ConvParams(kernel_size=3, stride=1, padding="same")
CONV_5X5_SAME = ConvParams(kernel_size=5, stride=1, padding="same")
CONV_7X7_SAME = ConvParams(kernel_size=7, stride=1, padding="same")
CONV_9X9_SAME = ConvParams(kernel_size=9, stride=1, padding="same")


@dataclass
class NeighborhoodAttentionParams:
    attention_window: int = 5
    attention_stride: int = 3


ATTENTION_5X5 = NeighborhoodAttentionParams(attention_window=5, attention_stride=1)
ATTENTION_11X11 = NeighborhoodAttentionParams(attention_window=11, attention_stride=1)
ATTENTION_13X13 = NeighborhoodAttentionParams(attention_window=13, attention_stride=1)
ATTENTION_17X17 = NeighborhoodAttentionParams(attention_window=17, attention_stride=1)


@dataclass
class HeadParams:
    conv_params: ConvParams
    attn_params: NeighborhoodAttentionParams
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    intermediate_channels: Optional[int] = None
    is_causal: bool = False
    _initializing: bool = True

    def __post_init__(self):
        if self.conv_params is None:
            raise ValueError("conv_params must be provided")
        self.out_channels = self.out_channels or self.in_channels
        self.conv_params.in_channels = self.conv_params.in_channels or self.in_channels
        self.conv_params.out_channels = self.conv_params.out_channels or self.intermediate_channels
        self._initializing = False

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if self._initializing:
            return True
        if name == 'in_channels':
            self.conv_params.in_channels = self.conv_params.in_channels or value
        elif name == 'intermediate_channels':
            self.conv_params.out_channels = value


@dataclass
class MultiHeadAttentionParams:
    num_heads: int
    head_params: List[HeadParams]
    final_conv_params: ConvParams
    scale_factor: int = 1
    intermediate_channels: Optional[int] = None
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None

    def __post_init__(self):
        if len(self.head_params) != self.num_heads:
            raise ValueError("Number of heads must match number of head params")
        self.intermediate_channels = self.intermediate_channels or self.in_channels // self.num_heads if self.in_channels is not None else None
        self.out_channels = self.out_channels or self.in_channels

        for head in self.head_params:
            head.in_channels = head.in_channels or self.in_channels
            head.intermediate_channels = head.intermediate_channels or self.intermediate_channels
            head.out_channels = head.out_channels or head.intermediate_channels or self.intermediate_channels
            head.conv_params.in_channels = head.conv_params.in_channels or head.in_channels
            head.conv_params.out_channels = head.conv_params.out_channels or head.intermediate_channels

        if self.final_conv_params is None:
            total_out_channels = sum(
                head.out_channels for head in self.head_params if head.out_channels is not None)
            self.final_conv_params = ConvParams(kernel_size=1, in_channels=total_out_channels,
                                                out_channels=self.out_channels)
        else:
            self.final_conv_params.in_channels = self.final_conv_params.in_channels or sum(
                head.out_channels for head in self.head_params if head.out_channels is not None)
            self.final_conv_params.out_channels = self.final_conv_params.out_channels or self.out_channels

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ['in_channels', 'intermediate_channels', 'out_channels']:
            for head in self.head_params:
                if name == 'in_channels':
                    head.in_channels = value
                elif name == 'intermediate_channels':
                    head.intermediate_channels = head.intermediate_channels or value
                elif name == 'out_channels':
                    head.out_channels = head.out_channels or value
                head.conv_params.in_channels = head.in_channels
                head.conv_params.out_channels = head.conv_params.out_channels or head.intermediate_channels
            if name == 'out_channels':
                self.final_conv_params.out_channels = self.final_conv_params.out_channels or value


def get_toplevel_encoder_multihead_attn_params(out_channels):
    if out_channels > 32 :
        factor = 8
    else:
        factor = 2
    head_3x3 = HeadParams(conv_params=deepcopy(CONV_3X3), intermediate_channels=out_channels//factor,
                          attn_params=deepcopy(ATTENTION_5X5))
    head_5x5 = HeadParams(conv_params=deepcopy(CONV_5X5), intermediate_channels=out_channels//factor,
                          attn_params=deepcopy(ATTENTION_11X11))
    head_7x7 = HeadParams(conv_params=deepcopy(CONV_7X7), intermediate_channels=out_channels//factor,
                          attn_params=deepcopy(ATTENTION_13X13))
    head_9x9 = HeadParams(conv_params=deepcopy(CONV_9X9), intermediate_channels=out_channels//factor,
                          attn_params=deepcopy(ATTENTION_17X17))
    return MultiHeadAttentionParams(num_heads=8,
                                    head_params=[head_3x3, head_3x3, head_5x5, head_5x5, head_7x7,
                                                 head_7x7, head_9x9, head_9x9],
                                    final_conv_params=deepcopy(CONV_3X3))


def get_deep_encoder_multihead_attn_params(out_channels):
    head_3x3_1 = HeadParams(conv_params=deepcopy(CONV_3X3_SAME), intermediate_channels=out_channels // 8,
                          attn_params=deepcopy(ATTENTION_5X5))
    head_3x3_2 = HeadParams(conv_params=deepcopy(CONV_3X3_SAME),
                          intermediate_channels=out_channels // 8,
                          attn_params=deepcopy(ATTENTION_11X11))
    head_5x5_1 = HeadParams(conv_params=deepcopy(CONV_5X5_SAME),
                          intermediate_channels=out_channels // 8,
                          attn_params=deepcopy(ATTENTION_13X13))
    head_5x5_2 = HeadParams(conv_params=deepcopy(CONV_5X5_SAME),
                          intermediate_channels=out_channels // 8,
                          attn_params=deepcopy(ATTENTION_17X17))
    return MultiHeadAttentionParams(num_heads=8,
                                    head_params=[head_3x3_1, head_3x3_1, head_3x3_2, head_3x3_2, head_5x5_1,
                                                 head_5x5_1, head_5x5_2, head_5x5_2],
                                    final_conv_params=deepcopy(CONV_3X3))


@dataclass
class TransformerParams:
    in_channels: int
    out_channels: int
    attention_params: Optional[MultiHeadAttentionParams] = None
    gated_residuals_params: Optional[ConvParams] = None
    final_conv_params: Optional[ConvParams] = None
    scale_factor: int = 1
    head_builder: Callable[
        [int], MultiHeadAttentionParams] = get_toplevel_encoder_multihead_attn_params

    def __post_init__(self):
        if not self.attention_params:
            self.attention_params = self.head_builder(self.out_channels)
        if not self.attention_params.in_channels:
            self.attention_params.in_channels = self.in_channels
        if not self.attention_params.out_channels:
            self.attention_params.out_channels = self.out_channels
        if not self.attention_params.intermediate_channels:
            self.attention_params.intermediate_channels = max(self.in_channels,
                                                              self.out_channels) // self.attention_params.num_heads
        self.attention_params.scale_factor = self.scale_factor
        if not self.final_conv_params:
            self.final_conv_params = deepcopy(CONV_1X1)
        if not self.final_conv_params.in_channels:
            self.final_conv_params.in_channels = self.attention_params.out_channels
        if not self.final_conv_params.out_channels:
            self.final_conv_params.out_channels = self.out_channels

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ['in_channels', 'out_channels', 'scale_factor'] and hasattr(self,
                                                                               'attention_params') and self.attention_params is not None:
            setattr(self.attention_params, name, value)
        if name == 'out_channels' and hasattr(self,
                                              'final_conv_params') and self.final_conv_params is not None:
            self.final_conv_params.out_channels = self.final_conv_params.out_channels or value
        if name == 'out_channel' and hasattr(self,
                                             'attention_params') and self.attention_params is not None:
            self.attention_params.out_channels = value


@dataclass
class GlobalAttentionParams:
    num_heads: int
    dropout: float = 0.1


@dataclass
class EncoderParams:
    transformer_params: List[TransformerParams]
    initial_conv_params: ConvParams
    num_global_attention_layers: int
    num_global_attention_heads: int = 16
    global_attention_dropout: float = 0.1


def build_encoder_params(in_channels: int, depths: List[int], scales: List[float],
                         num_global_attention_layers: int, num_global_attention_heads: int = 16,
                         global_attention_dropout: float = 0.1):
    initial_conv_params = ConvParams(kernel_size=3, in_channels=in_channels, out_channels=depths[0],
                                     groups=1, padding="same")
    transformer_params = []
    current_in_channels = depths[0]
    cummulative_scale = 1
    for index, depth, scale in zip(range(len(depths)), depths, scales):
        cummulative_scale *= scale
        head_builder = get_toplevel_encoder_multihead_attn_params if cummulative_scale <= 4 else get_deep_encoder_multihead_attn_params
        transformer_params.append(
            TransformerParams(in_channels=current_in_channels, out_channels=depth,
                              scale_factor=scale, head_builder=head_builder))
        current_in_channels = depth
    return EncoderParams(transformer_params=transformer_params,
                         initial_conv_params=initial_conv_params,
                         num_global_attention_heads=num_global_attention_heads,
                         num_global_attention_layers=num_global_attention_layers,
                         global_attention_dropout=global_attention_dropout)


DEFAULT_IMG_ENCODER_PARAMS = build_encoder_params(in_channels=3,
                                                  depths=[32, 32,32, 384, 768, 768],
                                                  scales=[1, 1, 1, 4, 4, 1],
                                                  num_global_attention_layers=4)
