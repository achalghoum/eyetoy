from copy import deepcopy
from dataclasses import dataclass
from math import gcd
from typing import Callable, List, Optional, Union


from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ConvParams:
    kernel_size: int
    stride: int = 1
    padding: Union[int,str] = "same"
    dilation: int = 1
    groups: int = 1
    bias: bool = True
    in_channels: Optional[int] = 0
    out_channels: Optional[int] = 0

@dataclass
class NeighborhoodAttentionParams:
    attention_window: int
    attention_stride: int

@dataclass
class HeadParams:
    attn_params: List[NeighborhoodAttentionParams]
    in_channels: int
    intermediate_channels: int
    conv_params: ConvParams
    dropout: float = 0.1

@dataclass
class MultiHeadAttentionParams:
    head_params: List[HeadParams]
    intermediate_channels: int
    out_channels: int
    final_conv_params: ConvParams
    scale_factor: int

@dataclass
class TransformerParams:
    in_channels: int
    out_channels: int
    attention_params: MultiHeadAttentionParams
    scale_factor: Optional[int] = 1

@dataclass
class GlobalAttentionParams:
    num_heads: int
    dropout: float = 0.1
@dataclass
class GlobalAttentionTransformerParams:
    d_model: int
    num_heads: int
    num_register_tokens: int = 4
    dropout: float = 0.1
    use_input_context_token: bool = False
    use_input_register_tokens: bool = False
@dataclass
class VisionTransformerParams:
    initial_conv_params: ConvParams
    transformer_params: List[TransformerParams]
    global_attention_params: GlobalAttentionTransformerParams
    input_channels: int
    output_channels: int
    num_classes: int

# Helper function to create ConvNATTransformerParams
def create_conv_nat_params(in_channels, out_channels,attn_kernel_sizes, conv_params, num_heads, scale_factor=1,min_intermediate_channels=32,conv_groups=32):
    head_params = []
    intermediate_channels = max(in_channels // len(conv_params) // num_heads,
                                min_intermediate_channels)
    for conv_params_1,attn_kernel_size in zip(conv_params, attn_kernel_sizes):
        conv_param = deepcopy(conv_params_1)
        conv_param.in_channels = in_channels
        conv_param.out_channels = intermediate_channels
        conv_param.groups = conv_groups if intermediate_channels % conv_groups == 0 else intermediate_channels
        head_params.append(HeadParams(
            attn_params=[
                NeighborhoodAttentionParams(attention_window=attn_kernel_size, attention_stride=1)]*num_heads,
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            conv_params=conv_param,
            dropout=0.1
        ))
    return TransformerParams(
        in_channels=in_channels,
        out_channels=out_channels,
        attention_params=MultiHeadAttentionParams(
            head_params=head_params,
            intermediate_channels=in_channels,
            out_channels=out_channels,
            final_conv_params=ConvParams(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding="same"
            ),
            scale_factor=scale_factor
        ),
        scale_factor=scale_factor
    )

first_size_convs = [ConvParams(kernel_size=3,stride=1,padding="same"),
                    ConvParams(kernel_size=3, stride=3,padding="valid"),
                    ConvParams(kernel_size=5, stride=5,padding="valid"),
                    ConvParams(kernel_size=7, stride=7,padding="valid")]
second_size_convs = [ConvParams(kernel_size=1,stride=1,padding="same"),
                     ConvParams(kernel_size=2,stride=2,padding="valid"),
                     ConvParams(kernel_size=3,stride=3,padding="valid"),
                     ConvParams(kernel_size=4,stride=4,padding="valid")]
final_size_convs = [ConvParams(kernel_size=1, stride=1,padding="same")]
first_attention_params = [17, 13, 11, 7]
second_attention_params = [11, 11, 11, 11]
third_attention_params = [7, 7, 7, 7]
final_attention_params = [5]
# Create the specific configuration
DEFAULT_IMG_ENCODER_PARAMS = VisionTransformerParams(
    initial_conv_params= ConvParams(kernel_size=3,padding="same",stride=1,in_channels=3,out_channels=32),
    transformer_params=[
        # First 4 layers with 32 channels
        create_conv_nat_params(32, 128,first_attention_params, first_size_convs, num_heads=4, scale_factor=2) ,
        create_conv_nat_params(128, 128,first_attention_params, first_size_convs, num_heads=4) ,
        # Downsampling layer to 368 channels
        create_conv_nat_params(128, 256,first_attention_params, first_size_convs, num_heads=4, scale_factor=2),
        create_conv_nat_params(256, 256,second_attention_params, second_size_convs, num_heads=4),

        # 4 layers with 368 channels
        create_conv_nat_params(256, 512,second_attention_params, second_size_convs, num_heads=4,scale_factor=2),
        create_conv_nat_params(512, 512,second_attention_params, second_size_convs, num_heads=4),
        # Downsampling layer to 768 channels
        create_conv_nat_params(512, 768,third_attention_params, second_size_convs, num_heads=4, scale_factor=2),

        # Final layer with 768 channels and 16 heads
        create_conv_nat_params(768, 768,final_attention_params, final_size_convs, num_heads=4),
    ],
    global_attention_params=GlobalAttentionTransformerParams(
        d_model=768,
        num_heads=16,
        num_register_tokens=4,
        dropout=0.1
    ),
    input_channels=3,
    output_channels=768,
    num_classes=1000  # Assuming 1000 classes, adjust as needed
)
