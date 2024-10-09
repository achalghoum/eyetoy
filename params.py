from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ConvParams:
    kernel_size: int
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    padding: str = "valid"
    bias: bool = True
    activation: str = "swiglu"
    dilation: int = 0
    stride: int = 1

CONV_1X1 = ConvParams(kernel_size=1, stride=1, padding="same")
CONV_3X3 = ConvParams(kernel_size=3, stride=1, padding="same")
CONV_5X5 = ConvParams(kernel_size=5, stride=2)
CONV_11X11 = ConvParams(kernel_size=11, stride=6)
CONV_17X17 = ConvParams(kernel_size=17, stride=17)



@dataclass
class NeighborhoodAttentionParams:
    min_kernel_size: int = 5
    max_kernel_size: int = 17
    min_dilation: int = 0
    max_dilation: int = 3

ATTENTION_3X3 = NeighborhoodAttentionParams(min_kernel_size=3, max_kernel_size=5, min_dilation=0, max_dilation=0)
ATTENTION_5X5 = NeighborhoodAttentionParams(min_kernel_size=7, max_kernel_size=13, min_dilation=0, max_dilation=2)
ATTENTION_11X11 = NeighborhoodAttentionParams(min_kernel_size=11, max_kernel_size=17, min_dilation=0, max_dilation=5)
ATTENTION_17X17 = NeighborhoodAttentionParams(min_kernel_size=13, max_kernel_size=19, min_dilation=0, max_dilation=7)


@dataclass
class HeadParams:
    conv_params: ConvParams
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None	
    attn_params: NeighborhoodAttentionParams = field(default_factory=NeighborhoodAttentionParams)
    intermediate_channels: Optional[int] = None
    is_causal: bool = False

    def __post_init__(self):
        if self.conv_params is None:
            raise ValueError("conv_params must be provided")
        if self.in_channels is None:
            raise ValueError("in_channels must be provided")
        self.intermediate_channels = self.intermediate_channels or self.in_channels
        self.out_channels = self.out_channels or self.in_channels
        self.conv_params.in_channels = self.conv_params.in_channels or self.in_channels
        self.conv_params.out_channels = self.conv_params.out_channels or self.intermediate_channels
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name == 'in_channels':
            self.conv_params.in_channels = self.conv_params.in_channels or value
        elif name == 'intermediate_channels':
            self.conv_params.out_channels = value


HEAD_3X3 = HeadParams(conv_params=CONV_3X3, intermediate_channels=32, attn_params=ATTENTION_3X3)
HEAD_5X5 = HeadParams(conv_params=CONV_5X5, intermediate_channels=64, attn_params=ATTENTION_5X5)
HEAD_11X11 = HeadParams(conv_params=CONV_11X11, intermediate_channels=128, attn_params=ATTENTION_11X11)
HEAD_17X17 = HeadParams(conv_params=CONV_17X17, intermediate_channels=256, attn_params=ATTENTION_17X17)

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
            head.out_channels = head.out_channels or self.intermediate_channels
            head.conv_params.in_channels = head.conv_params.in_channels or head.in_channels
            head.conv_params.out_channels = head.conv_params.out_channels or head.intermediate_channels

        if self.final_conv_params is None:
            total_out_channels = sum(head.out_channels for head in self.head_params if head.out_channels is not None)
            self.final_conv_params = ConvParams(kernel_size=1, in_channels=total_out_channels, out_channels=self.out_channels)
        else:
            self.final_conv_params.in_channels = self.final_conv_params.in_channels or sum(head.out_channels for head in self.head_params if head.out_channels is not None)
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
                head.conv_params.in_channels =  head.in_channels
                head.conv_params.out_channels = head.conv_params.out_channels or head.intermediate_channels
            if name == 'out_channels':
                self.final_conv_params.out_channels = self.final_conv_params.out_channels or value

DEFAULT_MULTIHEAD_ATTENTION = MultiHeadAttentionParams(num_heads=4,
                                                       head_params=[HEAD_3X3, HEAD_5X5, HEAD_11X11, HEAD_17X17],
                                                       final_conv_params=CONV_1X1)

DEFAULT_NUM_HEADS = 4

@dataclass
class TransformerParams:
    in_channels: int
    out_channels: int
    attention_params: Optional[MultiHeadAttentionParams] = DEFAULT_MULTIHEAD_ATTENTION
    final_conv_params: Optional[ConvParams] = CONV_1X1	
    use_gated_residuals: bool = True
    gated_residual_conv_params: Optional[ConvParams] = CONV_1X1
    scale_factor: int = 1

    def __post_init__(self):
        if not self.attention_params:
            self.attention_params = DEFAULT_MULTIHEAD_ATTENTION
        if not self.attention_params.in_channels:
            self.attention_params.in_channels = self.in_channels
        if not self.attention_params.out_channels:
            self.attention_params.out_channels = self.out_channels
        if not self.attention_params.intermediate_channels:
            self.attention_params.intermediate_channels = max(self.in_channels,self.out_channels)//self.attention_params.num_heads
        if not self.final_conv_params:
            self.final_conv_params = ConvParams(kernel_size=1,
                                                in_channels=self.attention_params.out_channels,
                                                out_channels=self.out_channels)
        if not self.final_conv_params.in_channels:
            self.final_conv_params.in_channels = self.attention_params.out_channels
        if not self.final_conv_params.out_channels:
            self.final_conv_params.out_channels = self.out_channels
        if self.use_gated_residuals and self.gated_residual_conv_params is None:
            self.gated_residual_conv_params = ConvParams(kernel_size=1,
                                                         in_channels=self.attention_params.out_channels + self.final_conv_params.out_channels,
                                                         out_channels= 1,
                                                         activation="sigmoid")

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in ['in_channels', 'out_channels', 'scale_factor'] and hasattr(self, 'attention_params'):
            setattr(self.attention_params, name, value)
        if name == 'out_channels' and hasattr(self, 'final_conv_params'):
            self.final_conv_params.out_channels = self.final_conv_params.out_channels or value
        if name == 'out_channels' and hasattr(self, 'gated_residual_conv_params'):
            self.gated_residual_conv_params.out_channels = self.gated_residual_conv_params.out_channels or value
        if name =='out_channel' and hasattr(self,'attention_params') and self.use_gated_residuals:
            self.attention_params.out_channels = value
        if self.use_gated_residuals and self.gated_residual_conv_params is None:
            self.gated_residual_conv_params = ConvParams(kernel_size=1,
                                                         in_channels=self.attention_params.out_channels + self.final_conv_params.out_channels,
                                                         out_channels= 1,
                                                         activation="sigmoid")


@dataclass
class GlobalAttentionParams:
    num_heads: int
    dropout: float = 0.1

@dataclass
class EncoderParams:
    transformer_params: List[TransformerParams]
    num_global_attention_heads: int
    global_attention_dropout: float
    num_global_attention_layers: int

    def __post_init__(self):
        for i in range(1, len(self.transformer_params)):
            prev_param = self.transformer_params[i-1]
            curr_param = self.transformer_params[i]
            curr_param.in_channels = prev_param.out_channels    