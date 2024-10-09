from abc import ABC
from typing import Type, Generic, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import Module, Conv2d, Conv3d, Conv1d

from attention import ConvNAT2D, ConvNAT3D, ConvNAT1D
from generics import ConvType, ConvNATType, ConvMultiHeadNATType
from params import HeadParams, ConvParams, MultiHeadAttentionParams
from positional_encoding import positional_encoding


class ConvMultiHeadNAT(ABC, Module, Generic[ConvType, ConvNATType]):
    conv_attn_type: Type[ConvNATType]
    conv_type: Type[ConvType]

    def __init__(self, num_heads: int, head_params: List[HeadParams], in_channels: int,
                 intermediate_channels: int, out_channels: int, final_conv_params: ConvParams,
                 scale_factor: float):
        super().__init__()
        self.attention_heads = torch.nn.ModuleList([
            self.conv_attn_type(**head_param.__dict__) for head_param in head_params
        ])
        self.final_conv = self.conv_type(**final_conv_params.__dict__)
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute output size based on scale factor
        input_size = x.shape[2:]  # Assuming NCHW or NCDHW format
        output_size = [int(i * self.scale_factor) for i in input_size]

        # Process all heads for all batches
        batch_size = x.shape[0]
        head_outputs = [head(x) for head in self.attention_heads]

        # Resize all head outputs to match the scaled input size
        resized_outputs = [torch.nn.functional.interpolate(output, size=output_size, mode='nearest')
                           for output in head_outputs]

        # Concatenate head outputs along the channel dimension for each batch
        combined_output = torch.cat(resized_outputs, dim=1)

        # Apply final convolution
        return self.final_conv(combined_output)


class ConvMultiHeadNAT1D(ConvMultiHeadNAT[Conv1d, ConvNAT1D]):
    conv_attn_type = ConvNAT1D
    conv_type = Conv1d


class ConvMultiHeadNAT2D(ConvMultiHeadNAT[Conv2d, ConvNAT2D]):
    conv_attn_type = ConvNAT2D
    conv_type = Conv2d


class ConvMultiHeadNAT3D(ConvMultiHeadNAT[Conv3d, ConvNAT3D]):
    conv_attn_type = ConvNAT3D
    conv_type = Conv3d


class ConvNATTransformer(ABC, Module, Generic[ConvType, ConvMultiHeadNATType]):
    multi_head_attention_type: Type[ConvMultiHeadNATType]
    conv_type: Type[ConvType]

    def __init__(self, attention_params: MultiHeadAttentionParams, final_conv_params: ConvParams,
                 use_gated_residuals: bool = False, gated_residuals_params: ConvParams = None):
        super(Module).__init__()
        self.multi_head_attention = self.multi_head_attention_type(**attention_params.__dict__)
        self.final_conv = self.conv_type(**final_conv_params.__dict__)
        self.use_gated_residuals = use_gated_residuals
        if use_gated_residuals:
            self.gated_residuals = self.conv_type(**gated_residuals_params.__dict__)
        layernorm_channels = final_conv_params.out_channels if not use_gated_residuals else gated_residuals_params.out_channels
        self.layernorm = torch.nn.LayerNorm(layernorm_channels)

    def residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.use_gated_residuals:
            weights = self.gated_residuals(torch.cat([x, y], dim=-1))
            return (1 - weights) * x + weights * y
        else:
            return x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = self.multi_head_attention(x)
        conv_output = self.final_conv(attention_output)
        residual = self.residual(attention_output, conv_output)
        return self.layernorm(residual)


class ConvNATTransformer1D(ConvNATTransformer[Conv1d, ConvMultiHeadNAT1D]):
    multi_head_attention_type = ConvMultiHeadNAT1D
    conv_type = Conv1d


class ConvNATTransformer2D(ConvNATTransformer[Conv2d, ConvMultiHeadNAT2D]):
    multi_head_attention_type = ConvMultiHeadNAT2D
    conv_type = Conv2d


class ConvNATTransformer3D(ConvNATTransformer[Conv3d, ConvMultiHeadNAT3D]):
    multi_head_attention_type = ConvMultiHeadNAT3D
    conv_type = Conv3d


class GlobalAttentionBlock(Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_input_context_token: bool = False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_input_context_token = use_input_context_token
        if not use_input_context_token:
            self.context_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                               batch_first=True)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, context_token: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # x shape: (batch_size, channels, height, width) or (batch_size, channels, depth, height, width)
        # context_token shape (if provided): (batch_size, 1, d_model)
        batch_size = x.shape[0]
        original_shape = x.shape
        pe = positional_encoding(x)
        # Add positional encoding
        x = x + pe
        # Flatten the spatial dimensions
        x_flat = x.flatten(2).transpose(1, 2)  # (batch_size, H*W or D*H*W, channels)

        # Use input context token or learned parameter
        if self.use_input_context_token:
            if context_token is None:
                raise ValueError("context token expected but not provided")
        else:
            context_token = self.context_token.expand(batch_size, -1, -1)

        # Append context token to the flattened feature map
        x_with_context = torch.cat([context_token, x_flat], dim=1)

        # Create attention mask
        seq_len = x_with_context.size(1)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        mask[0, :] = False  # Context token can attend to all tokens
        mask[1:, 1:] = True  # Other tokens cannot attend to each other

        if self.use_input_context_token:
            mask[1:, 0] = False  # All tokens can attend to the context token
        else:
            mask[1:, 0] = True  # Unless we are at the first layer 

        # Apply self-attention with mask
        attn_output, _ = self.attention(x_with_context, x_with_context, x_with_context,
                                        attn_mask=mask)

        # Apply layer normalization
        output = self.layernorm(attn_output + x_with_context)

        # Split context token and feature map
        context_token_out = output[:, 0:1, :]
        feature_map_out = output[:, 1:, :]

        # Reshape feature map to original shape
        feature_map_out = feature_map_out.transpose(1, 2).reshape(original_shape)

        return feature_map_out, context_token_out


