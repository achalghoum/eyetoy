from abc import ABC
from typing import Type, Generic, Optional, Tuple

import torch
from torch import nn
from torch.nn import Module, Conv2d, Conv3d, Conv1d, LayerNorm, Dropout
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, or_masks
from torch.nn.functional import interpolate

from generics import ConvType, MultiScaleMultiHeadNAType
from layers.attention import MulitScaleMultiHeadNA1D, MulitScaleMultiHeadNA2D, MulitScaleMultiHeadNA3D
from params import ConvParams, MultiHeadAttentionParams
from .norms import LayerNorm2d, LayerNorm3d
from .positional_encoding import positional_encoding

class Swish(Module):
    def forward(self,x):
        return x * torch.sigmoid(x)
 

class MSNATTransformer(ABC, Module, Generic[ConvType, MultiScaleMultiHeadNAType]):
    multi_head_attention_type: Type[MultiScaleMultiHeadNAType]
    conv_type: Type[ConvType]
    norm_type: Type[Module]

    def __init__(self, in_channels: int, out_channels: int,
                 attention_params: MultiHeadAttentionParams,
                 scale_factor: Optional[float] = 1.,
                 dropout=0.2, **kwargs):
        super(MSNATTransformer, self).__init__()
        self.multi_head_attention = self.multi_head_attention_type(
            **attention_params.__dict__)
        self.final_conv = nn.Sequential(self.conv_type(kernel_size=1,
                                                       in_channels=out_channels,
                                                       out_channels=out_channels*4),
                                        Swish(),
                                        self.conv_type(kernel_size=1,
                                                       in_channels=out_channels * 4,
                                                       out_channels=out_channels)
                                        )
        self.scale_factor = scale_factor or attention_params.scale_factor
        self.layernorm1 = self.norm_type(in_channels)
        self.layernorm2 = self.norm_type(out_channels)
        self.dropout = Dropout(dropout)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._set_shortcut()
        self._set_rescale()

    def _set_shortcut(self):
        self.shortcut = self.conv_type(in_channels=self.in_channels, out_channels=self.out_channels,
                                       kernel_size=1,
                                       padding="same") if self.in_channels != self.out_channels else nn.Identity()
    def _set_rescale(self):
        if self.scale_factor != 1:
            self.rescale = lambda x: interpolate(x, scale_factor= self.scale_factor,
                                         mode="bilinear")
        else:
            self.rescale = lambda x: x

    def residual_downsample(self, x: torch.Tensor) -> torch.Tensor:
        # Handle different channel dimensions using addition list in residual connections
        return self.shortcut(self.rescale(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_downsample(x) + self.multi_head_attention(self.dropout(self.layernorm1(x)))
        x = x + self.final_conv(self.dropout(self.layernorm2(x)))
        return x


class MSNATTransformer1D(MSNATTransformer[Conv1d, MulitScaleMultiHeadNA1D]):
    multi_head_attention_type = MulitScaleMultiHeadNA1D
    conv_type = Conv1d
    norm_type = LayerNorm


class MSNATTransformer2D(MSNATTransformer[Conv2d, MulitScaleMultiHeadNA2D]):
    multi_head_attention_type = MulitScaleMultiHeadNA2D
    conv_type = Conv2d
    norm_type = LayerNorm2d


class MSNATTransformer3D(MSNATTransformer[Conv3d, MulitScaleMultiHeadNA3D]):
    multi_head_attention_type = MulitScaleMultiHeadNA3D
    conv_type = Conv3d
    norm_type = LayerNorm3d


class GlobalAttentionTransformer(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_register_tokens: int = 4,
            dropout: float = 0.1,
            use_input_context_token: bool = False,
            use_input_register_tokens: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_register_tokens = num_register_tokens
        self.use_input_context_token = use_input_context_token
        self.use_input_register_tokens = use_input_register_tokens

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.head_dim = d_model // num_heads

        if not use_input_context_token:
            self.context_token = nn.Parameter(torch.randn(1, 1, d_model))
        if not use_input_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, d_model))

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            Swish(),
            nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

        # Cached mask
        self.cached_mask = None
        self.cached_mask_params = None

    def create_attention_mask(self, batch_size: int, seq_len: int,
                              device: str = "cuda"):
        """Creates and caches the attention mask."""
        # Check if the mask is already cached
        if self.cached_mask is not None:
            if self.cached_mask_params == (batch_size, seq_len, device):
                return self.cached_mask

        def q_mask(batch, head, q_idx, kv_idx):
            return q_idx < self.num_register_tokens+1
        def kv_mask(batch, head, q_idx, kv_idx):
            return kv_idx < self.num_register_tokens+1

        def self_mask(batch, head, q_idx, kv_idx):
            return q_idx == kv_idx

        block_mask = create_block_mask(
            or_masks(q_mask,kv_mask,self_mask),
            B=batch_size,
            H=self.num_heads,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device
        )

        # Cache the mask
        self.cached_mask = block_mask
        self.cached_mask_params = (batch_size, seq_len, device)

        return block_mask

    def reshape_to_heads(self, tensor, batch_size):
        """Reshapes tensor for multi-head attention."""
        return tensor.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def reshape_from_heads(self, tensor, batch_size, seq_len):
        """Reshapes tensor back from multi-head attention."""
        return tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def initialize_tokens(self, batch_size: int, context_token: Optional[torch.Tensor],
                          register_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        """Initialize context and register tokens."""
        # Handle context token
        if self.use_input_context_token:
            if context_token is None:
                raise ValueError("Context token expected but not provided")
        else:
            context_token = self.context_token.expand(batch_size, -1, -1)

        # Handle register tokens
        if self.use_input_register_tokens:
            if register_tokens is None:
                raise ValueError("Register tokens expected and not provided")
        else:
            register_tokens = self.register_tokens.expand(batch_size, -1, -1)

        # Concatenate tokens
        tokens = torch.cat([context_token, register_tokens], dim=1)

        # Add positional encoding if not using input context token
        if not self.use_input_context_token:
            tokens = tokens + positional_encoding(tokens, d=self.d_model)

        return tokens

    def forward(
            self,
            x: torch.Tensor,
            context_token: Optional[torch.Tensor] = None,
            register_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        original_shape = x.shape

        x_flat = x.flatten(2).transpose(1, 2)

        tokens = self.initialize_tokens(batch_size, context_token, register_tokens)
        x_w_tokens = self.layernorm1(torch.cat([tokens, x_flat], dim=1))

        queries = self.reshape_to_heads(self.query_proj(x_w_tokens), batch_size)
        keys = self.reshape_to_heads(self.key_proj(x_w_tokens), batch_size)
        values = self.reshape_to_heads(self.value_proj(x_w_tokens), batch_size)

        seq_len = x_w_tokens.size(1)
        block_mask = self.create_attention_mask(batch_size, seq_len)

        attn_output = flex_attention(
            query=queries,
            key=keys,
            value=values,
            block_mask=block_mask
        )

        attn_output = self.reshape_from_heads(attn_output, batch_size, seq_len)
        attn_output = attn_output + x_w_tokens

        attn_output = self.ffn(self.dropout(self.layernorm2(attn_output))) + attn_output

        context_token_out = attn_output[:, 0:1, :]
        register_tokens_out = attn_output[:, 1:1 + self.num_register_tokens, :]
        feature_map_out = attn_output[:, 1 + self.num_register_tokens:, :]
        feature_map_out = feature_map_out.transpose(1, 2).reshape(original_shape)

        return feature_map_out, context_token_out, register_tokens_out

