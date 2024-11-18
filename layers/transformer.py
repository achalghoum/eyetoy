from abc import ABC
from typing import Type, Generic, Optional, Tuple

import torch
from torch import nn
from torch.nn import Module, Conv2d, Conv3d, Conv1d, LayerNorm, Dropout
from torch.nn.functional import interpolate

from generics import ConvType, MultiScaleMultiHeadNAType
from layers.attention import MulitScaleMultiHeadNA1D, MulitScaleMultiHeadNA2D, MulitScaleMultiHeadNA3D
from params import ConvParams, MultiHeadAttentionParams
from .norms import LayerNorm2d, LayerNorm3d
from .positional_encoding import positional_encoding


class MSNATTransformer(ABC, Module, Generic[ConvType, MultiScaleMultiHeadNAType]):
    multi_head_attention_type: Type[MultiScaleMultiHeadNAType]
    conv_type: Type[ConvType]
    norm_type: Type[Module]

    def __init__(self, in_channels: int, out_channels: int,
                 attention_params: MultiHeadAttentionParams,
                 scale_factor: Optional[float] = 1., **kwargs):
        super(MSNATTransformer, self).__init__()
        self.multi_head_attention = self.multi_head_attention_type(
            **attention_params.__dict__)
        self.final_conv = nn.Sequential(self.conv_type(kernel_size=1,
                                                       in_channels=out_channels,
                                                       out_channels=out_channels*4),
                                        nn.GELU(),
                                        self.conv_type(kernel_size=1,
                                                       in_channels=out_channels * 4,
                                                       out_channels=out_channels)
                                        )
        self.scale_factor = scale_factor or attention_params.scale_factor
        self.layernorm1 = self.norm_type(in_channels)
        self.layernorm2 = self.norm_type(out_channels)
        self.dropout = Dropout(0.2)
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

        # Initialize learnable tokens if not provided as input
        if not use_input_context_token:
            self.context_token = nn.Parameter(torch.randn(1, 1, d_model))
        if not use_input_register_tokens:
            self.register_tokens = nn.Parameter(torch.randn(1, num_register_tokens, d_model))

        # Core transformer components
        self.attention = nn.MultiheadAttention(
            d_model, 
            num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network with expansion factor of 4
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def create_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Creates the attention mask for the transformer."""
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        
        # Context token can attend to all tokens
        mask[0, :] = False
        
        # Register tokens can attend to all tokens
        mask[1:1 + self.num_register_tokens, :] = False
        
        # All tokens can attend to context and register tokens
        mask[1 + self.num_register_tokens:, :1 + self.num_register_tokens] = not self.use_input_context_token
        
        # Create diagonal mask for remaining tokens
        mask[1 + self.num_register_tokens:, 1 + self.num_register_tokens:] = ~torch.eye(mask.size(0) - (1 + self.num_register_tokens), dtype=torch.bool, device=mask.device)
        return mask

    def initialize_tokens(self, batch_size: int, context_token: Optional[torch.Tensor], register_tokens: Optional[torch.Tensor]) -> torch.Tensor:
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
        batch_size = x.shape[0]
        original_shape = x.shape
        
        # Flatten spatial dimensions
        x_flat = x.flatten(2).transpose(1, 2)

        # Initialize tokens
        tokens = self.initialize_tokens(batch_size, context_token, register_tokens)

        # Concatenate tokens with input and apply layer norm
        x_w_tokens = self.layernorm1(torch.cat([tokens, x_flat], dim=1))
        
        # Create attention mask
        mask = self.create_attention_mask(x_w_tokens.size(1), x.device)

        # Multi-head attention
        attn_output, _ = self.attention(
            query=x_w_tokens,
            key=x_w_tokens,
            value=x_w_tokens,
            attn_mask=mask
        )

        # First residual connection
        attn_output = attn_output + x_w_tokens
        
        # FFN and second residual connection
        attn_output = self.ffn(self.layernorm2(attn_output)) + attn_output

        # Split output into components
        context_token_out = attn_output[:, 0:1, :]
        register_tokens_out = attn_output[:, 1:1 + self.num_register_tokens, :]
        feature_map_out = attn_output[:, 1 + self.num_register_tokens:, :]
        
        # Reshape feature map back to original dimensions
        feature_map_out = feature_map_out.transpose(1, 2).reshape(original_shape)

        return feature_map_out, context_token_out, register_tokens_out
