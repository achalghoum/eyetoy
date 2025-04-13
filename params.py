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
    num_heads: int
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
def create_ms_nat_params(
    in_channels,
    out_channels,
    attn_kernel_sizes,
    conv_params,
    num_heads,
    scale_factor=1.0,
    dropout=0.2,
    conv_groups=32,
):
    head_params = []
    intermediate_channels = out_channels // len(conv_params) // num_heads
    for conv_params_1, attn_kernel_size in zip(conv_params, attn_kernel_sizes):
        conv_param = deepcopy(conv_params_1)
        conv_param.in_channels = in_channels
        conv_param.out_channels = out_channels
        conv_param.groups = (
            conv_groups if in_channels % conv_groups == 0 else in_channels
        )
        head_params.append(
            HeadParams(
                attn_params=NeighborhoodAttentionParams(
                    attention_window=attn_kernel_size, attention_stride=1
                ),
                num_heads=num_heads,
                in_channels=in_channels,
                intermediate_channels=intermediate_channels,
                conv_params=conv_param,
                dropout=dropout,
            )
        )
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
                padding="same",
            ),
            scale_factor=scale_factor,
        ),
        scale_factor=scale_factor,
    )


first_size_convs = [
    ConvParams(kernel_size=1, stride=1, padding="same"),
    ConvParams(kernel_size=2, stride=2, padding="valid"),
    ConvParams(kernel_size=4, stride=4, padding="valid"),
    ConvParams(kernel_size=8, stride=8, padding="valid"),
]
second_size_convs = [
    ConvParams(kernel_size=1, stride=1, padding="same"),
    ConvParams(kernel_size=2, stride=2, padding="valid"),
    ConvParams(kernel_size=3, stride=3, padding="valid"),
    ConvParams(kernel_size=4, stride=4, padding="valid"),
]
final_size_convs = [ConvParams(kernel_size=1, stride=1, padding="same")]
first_attention_params = [17, 13, 11, 7]
second_attention_params = [7, 5, 3, 3]
final_attention_params = [7]
# Create the specific configuration
DEFAULT_IMG_ENCODER_PARAMS = EncoderParams(
    initial_conv_params=ConvParams(
        kernel_size=3, padding="same", stride=1, in_channels=3, out_channels=64,
    ),
    transformer_params=[
        create_ms_nat_params(
            64,
            128,
            first_attention_params,
            first_size_convs,
            num_heads=2,
            scale_factor=0.5,
        ),
        create_ms_nat_params(
            128,
            128,
            first_attention_params,
            first_size_convs,
            num_heads=2,
            scale_factor=1,
        ),
        create_ms_nat_params(
            128,
            256,
            first_attention_params,
            first_size_convs,
            num_heads=2,
            scale_factor=0.5,
        ),
        create_ms_nat_params(
            256,
            256,
            first_attention_params,
            first_size_convs,
            num_heads=2,
            scale_factor=1,
        ),
        create_ms_nat_params(
            256,
            512,
            second_attention_params,
            second_size_convs,
            num_heads=2,
            scale_factor=0.5,
        ),
        create_ms_nat_params(
            512,
            512,
            second_attention_params,
            second_size_convs,
            num_heads=2,
            scale_factor=1,
        ),
        create_ms_nat_params(
            512,
            1024,
            second_attention_params,
            second_size_convs,
            num_heads=2,
            scale_factor=0.5,
        ),
        create_ms_nat_params(
            1024,
            1024,
            final_attention_params,
            final_size_convs,
            num_heads=8,
            scale_factor=1,
        ),
    ],
    global_attention_params=GlobalAttentionTransformerParams(
        d_model=512, num_heads=8, num_layers=4, num_register_tokens=32, dropout=0.2
    ),
    input_channels=3,
    output_channels=512,
)


def create_simple_transformer_params(
    in_channels: int,
    out_channels: int,
    attention_window: int,
    num_heads: int,
    kernel_size: int = 3,
    attention_stride: int = 1,
    dropout: float = 0.1,
    scale_factor: float = 1.0,
    is_multihead: bool = False,
) -> TransformerParams:
    """
    Create a TransformerParams instance with simplified parameters.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        attention_window: Size of the attention window
        num_heads: Number of attention heads
        kernel_size: Kernel size for convolutions
        attention_stride: Stride for attention
        dropout: Dropout rate
        scale_factor: Scale factor
        is_multihead: If True, uses kernel_size=1 for convolutions

    Returns:
        A TransformerParams instance
    """
    # Determine intermediate channels based on heads
    intermediate_channels = out_channels // num_heads

    # Create head parameters
    head_params = [
        HeadParams(
            attn_params=NeighborhoodAttentionParams(
                attention_window=attention_window, attention_stride=attention_stride
            ),
            num_heads=num_heads,
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            conv_params=ConvParams(
                kernel_size=1 if is_multihead else kernel_size,
                stride=1,
                padding="same",
                in_channels=in_channels,
                out_channels=out_channels,
            ),
            dropout=dropout,
        )
    ]

    # Create and return TransformerParams
    return TransformerParams(
        in_channels=in_channels,
        out_channels=out_channels,
        attention_params=MultiHeadAttentionParams(
            head_params=head_params,
            intermediate_channels=intermediate_channels,
            out_channels=out_channels,
            final_conv_params=ConvParams(
                kernel_size=1,
                stride=1,
                padding="same",
                in_channels=in_channels,
                out_channels=out_channels,
            ),
            scale_factor=scale_factor,
        ),
        scale_factor=scale_factor,
    )


def create_diffusion_transformer_params(
    dim: int,
    channels: List[int],
    window_sizes: List[int] = [7, 5, 3, 3],
    num_heads_multiplier: int = 4,
) -> List[List[TransformerParams]]:
    """
    Create transformer parameters for each level of the diffusion U-Net.

    Args:
        dim: Dimensionality (1, 2, or 3)
        channels: List of channel counts for each level
        window_sizes: List of attention window sizes for each level
        num_heads_multiplier: Multiplier for determining number of heads

    Returns:
        List of transformer parameter lists for each level
    """
    transformer_params_per_level = []

    for i, channel_count in enumerate(channels):
        num_heads = channel_count // num_heads_multiplier
        window_size = window_sizes[i] if i < len(window_sizes) else 3

        # For each level, create two types of attention blocks
        level_params = [
            # Neighborhood attention
            create_simple_transformer_params(
                in_channels=channel_count,
                out_channels=channel_count,
                attention_window=window_size,
                num_heads=num_heads,
                is_multihead=False,
            ),
            # Global/multihead attention
            create_simple_transformer_params(
                in_channels=channel_count,
                out_channels=channel_count,
                attention_window=1,
                num_heads=num_heads,
                is_multihead=True,
            ),
        ]
        transformer_params_per_level.append(level_params)

    return transformer_params_per_level
