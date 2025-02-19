from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional
from typing import Union


@dataclass
class ConvParams:
    kernel_size: int
    stride: int = 1
    padding: Union[int, str] = "same"
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
    attn_params: NeighborhoodAttentionParams
    num_heads:int
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
    scale_factor: float


@dataclass
class TransformerParams:
    in_channels: int
    out_channels: int
    attention_params: MultiHeadAttentionParams
    scale_factor: Optional[float] = 1
    dropout: Optional[float] = 0.2


@dataclass
class GlobalAttentionParams:
    num_heads: int
    dropout: float = 0.1


@dataclass
class GlobalAttentionTransformerParams:
    d_model: int
    num_heads: int
    num_layers: int
    num_register_tokens: int = 4
    dropout: float = 0.1


@dataclass
class EncoderParams:
    initial_conv_params: ConvParams
    transformer_params: List[TransformerParams]
    global_attention_params: GlobalAttentionTransformerParams
    input_channels: int
    output_channels: int


# Helper function to create ConvNATTransformerParams
def create_ms_nat_params(in_channels, out_channels, attn_kernel_sizes, conv_params, num_heads,
                           scale_factor=1., dropout=0.2, conv_groups=32):
    head_params = []
    intermediate_channels = out_channels // len(conv_params) // num_heads
    for conv_params_1, attn_kernel_size in zip(conv_params, attn_kernel_sizes):
        conv_param = deepcopy(conv_params_1)
        conv_param.in_channels = in_channels
        conv_param.out_channels = out_channels
        conv_param.groups = conv_groups if in_channels % conv_groups == 0 else in_channels
        head_params.append(HeadParams(
            attn_params=NeighborhoodAttentionParams(attention_window=attn_kernel_size,attention_stride=1),
            num_heads = num_heads,
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            conv_params=conv_param,
            dropout=dropout
        ))
    return TransformerParams(
        in_channels=in_channels,
        out_channels=out_channels,
        attention_params=MultiHeadAttentionParams(
            head_params=head_params,
            intermediate_channels=intermediate_channels,
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


first_size_convs = [ConvParams(kernel_size=1, stride=1, padding="same"),
                    ConvParams(kernel_size=2, stride=2, padding="valid"),
                    ConvParams(kernel_size=4, stride=4, padding="valid"),
                    ConvParams(kernel_size=8, stride=8, padding="valid")]
second_size_convs = [ConvParams(kernel_size=1, stride=1, padding="same"),
                     ConvParams(kernel_size=2, stride=2, padding="valid"),
                     ConvParams(kernel_size=3, stride=3, padding="valid"),
                     ConvParams(kernel_size=4, stride=4, padding="valid")]
final_size_convs = [ConvParams(kernel_size=1, stride=1, padding="same")]
first_attention_params = [17, 13, 11, 7]
second_attention_params = [7, 5, 3, 3]
final_attention_params = [7]
# Create the specific configuration
DEFAULT_IMG_ENCODER_PARAMS = EncoderParams(
    initial_conv_params=ConvParams(kernel_size=3, padding="same", stride=1, in_channels=3,
                                   out_channels=32),
    transformer_params=[
        create_ms_nat_params(32, 64, first_attention_params, first_size_convs, num_heads=2,
                               scale_factor=0.5),
        create_ms_nat_params(64, 64, first_attention_params, first_size_convs, num_heads=2,
                               scale_factor=1),
        create_ms_nat_params(64, 128, first_attention_params, first_size_convs, num_heads=2,
                               scale_factor=0.5),
        create_ms_nat_params(128, 128, first_attention_params, first_size_convs, num_heads=2,
                               scale_factor=1),
        create_ms_nat_params(128, 256, second_attention_params, second_size_convs, num_heads=2,
                               scale_factor=0.5),
        create_ms_nat_params(256, 256, second_attention_params, second_size_convs, num_heads=2,
                               scale_factor=1),
        create_ms_nat_params(256, 512, second_attention_params, second_size_convs, num_heads=2,
                               scale_factor=0.5),
        create_ms_nat_params(512, 512, final_attention_params,final_size_convs , num_heads=8,
                               scale_factor=1)
    ],
    global_attention_params=GlobalAttentionTransformerParams(
        d_model=512,
        num_heads=8,
        num_layers = 4,
        num_register_tokens=32,
        dropout=0.2
    ),
    input_channels=3,
    output_channels=512,
)
