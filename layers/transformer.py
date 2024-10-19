from abc import ABC
from typing import Type, Generic, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module, Conv2d, Conv3d, Conv1d, InstanceNorm3d, LayerNorm
from torch.nn import functional as F

from generics import ConvType, ConvMultiHeadNATType
from layers.attention import ConvMultiHeadNAT1D, ConvMultiHeadNAT2D, ConvMultiHeadNAT3D
from params import ConvParams, MultiHeadAttentionParams
from .positional_encoding import positional_encoding


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape,
                         self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class ConvNATTransformer(ABC, Module, Generic[ConvType, ConvMultiHeadNATType]):
    multi_head_attention_type: Type[ConvMultiHeadNATType]
    conv_type: Type[ConvType]
    norm_type: Type[Module]

    def __init__(self, in_channels: int, out_channels: int,
                 attention_params: MultiHeadAttentionParams, final_conv_params: ConvParams,
                 dropout: float = 0.1, scale_factor: Optional[float] = 1, **kwargs):
        super(ConvNATTransformer, self).__init__()
        self.multi_head_attention = self.multi_head_attention_type(
            **attention_params.__dict__)
        self.final_conv = nn.Sequential(self.conv_type(**final_conv_params.__dict__),
                                        nn.GELU())
        if in_channels != out_channels:
            self.use_residual_scale = True
            self.residual_upscale = self.conv_type(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        else:
            self.use_residual_scale = False
        self.layernorm1 = self.norm_type(out_channels)
        self.layernorm2 = self.norm_type(out_channels)

        self.scale_factor = scale_factor or attention_params.scale_factor

    def residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Handle different channel dimensions using addition list in residual connections
        if self.use_residual_scale:
            x = self.residual_upscale(x)
        return x + y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = self.multi_head_attention(x)
        rescaled_input = torch.nn.functional.interpolate(
            x, size=attention_output.shape[2:], mode='nearest') if self.scale_factor != 1 else x
        residual_1 = self.layernorm1(
            self.residual(rescaled_input, attention_output))
        residual_2 = self.final_conv(residual_1)
        residual = self.layernorm2(residual_2 + residual_1)
        return residual


class ConvNATTransformer1D(ConvNATTransformer[Conv1d, ConvMultiHeadNAT1D]):
    multi_head_attention_type = ConvMultiHeadNAT1D
    conv_type = Conv1d
    norm_type = LayerNorm


class ConvNATTransformer2D(ConvNATTransformer[Conv2d, ConvMultiHeadNAT2D]):
    multi_head_attention_type = ConvMultiHeadNAT2D
    conv_type = Conv2d
    norm_type = LayerNorm2d


class ConvNATTransformer3D(ConvNATTransformer[Conv3d, ConvMultiHeadNAT3D]):
    multi_head_attention_type = ConvMultiHeadNAT3D
    conv_type = Conv3d
    norm_type = InstanceNorm3d


class GlobalAttentionBlock(Module):

    def __init__(self, d_model: int, num_heads: int, num_register_tokens: int = 4,
                 dropout: float = 0.1, use_input_context_token: bool = False,
                 use_input_register_tokens: bool = False):
        super(GlobalAttentionBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.use_input_context_token = use_input_context_token
        self.use_input_register_tokens = use_input_register_tokens
        if not use_input_context_token:
            self.context_token = nn.Parameter(torch.randn(1, 1, d_model))
        if not use_input_register_tokens:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_register_tokens, d_model))

        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                               batch_first=True)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        # Add feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context_token: Optional[torch.Tensor] = None,
                register_tokens: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        original_shape = x.shape
        x_flat = (x).flatten(2).transpose(1, 2)
        if self.use_input_context_token:
            if context_token is None:
                raise ValueError("Context token expected but not provided")
        else:
            context_token = self.context_token.expand(batch_size, -1, -1)

        if self.use_input_context_token:
            if register_tokens is None:
                raise ValueError("Register tokens expected and not provided")
        else:
            register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        tokens = torch.cat(
            [context_token, register_tokens], dim=1)
        if not self.use_input_context_token:
            tokens = tokens + positional_encoding(tokens, d=self.d_model)
        x_w_tokens = torch.cat(
            [tokens, x_flat], dim=1)

        seq_len = x_w_tokens.size(1)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        mask[0, :] = False  # Context token can attend to all tokens
        # Register tokens can attend to all tokens
        mask[1:1 + self.num_register_tokens, :] = False
        # All tokens can attend to context and register tokens
        mask[1 + self.num_register_tokens:,
        :1 + self.num_register_tokens] = not self.use_input_context_token
        # Other tokens cannot attend to each other
        mask[1 + self.num_register_tokens:,
        1 + self.num_register_tokens:] = self.use_input_context_token
        attn_output, _ = self.attention(
            key=x_w_tokens, query=x_w_tokens, value=x_w_tokens, attn_mask=mask)
        # Apply first residual connection and layer normalization
        attn_output = self.layernorm1(attn_output + x_w_tokens)

        # Apply feed-forward network
        ffn_output = self.ffn(attn_output)
        ffn_output = self.dropout(ffn_output)

        # Apply second residual connection and layer normalization
        output = self.layernorm2(ffn_output + attn_output)

        context_token_out = output[:, 0:1, :]
        register_tokens_out = output[:, 1:1 + self.num_register_tokens, :]
        feature_map_out = output[:, 1 + self.num_register_tokens:, :]

        feature_map_out = feature_map_out.transpose(
            1, 2).reshape(original_shape)

        return feature_map_out, context_token_out, register_tokens_out
