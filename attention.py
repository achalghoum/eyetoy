from torch.nn import Module, Conv2d, Conv3d, Parameter
import torch
from typing import TypeVar, Generic, List, Callable, Type
from positional_encoding import positional_encoding
from generics import ConvType
from natten.functional import na2d,na3d, na2d_qk, na3d_qk, na2d_av, na3d_av
from abc import ABC
from params import ConvParams, NeighborhoodAttentionParams

def check_conv_output_sizes(conv_params_list: List[ConvParams]) -> bool:
    output_shapes = [(2*param.padding-param.kernel_size)//param.stride+1 for param in conv_params_list]
    return len(set(output_shapes)) == 1


class ConvNAT(ABC, Module, Generic[ConvType]):
    atten_func: Callable
    qk_func: Callable
    v_func: Callable
    conv_type: Type[ConvType]
    def __init__(self, query_params: ConvParams,
                 key_params: ConvParams,
                 value_params: ConvParams,
                 attn_params: NeighborhoodAttentionParams,
                 in_channels: int,
                 out_channels: int,
                 intermediate_channels: int,
                 is_causal: bool = False
                 ):
        super().__init__()
        self.validate_params(query_params, key_params, value_params)
        self.query_conv = self.conv_type(**query_params.__dict__, in_channels=in_channels,out_channels=intermediate_channels)
        self.key_conv = self.conv_type(**key_params.__dict__, in_channels=in_channels,out_channels=intermediate_channels)
        self.value_conv = self.conv_type(**value_params.__dict__,in_channels=in_channels, out_channels=out_channels)
        self.dilation = Parameter(torch.tensor(attn_params.min_dilation, dtype=torch.int32), requires_grad=True)
        self.kernel_size = Parameter(torch.tensor(attn_params.min_kernel_size, dtype=torch.int32), requires_grad=True)
        self.min_dilation = attn_params.min_dilation
        self.max_dilation = attn_params.max_dilation
        self.min_kernel_size = attn_params.min_kernel_size
        self.max_kernel_size = attn_params.max_kernel_size
        self.is_causal = is_causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dilation = self.get_dilation()
        kernel_size = self.get_kernel_size()
        query = self.query_conv(x)
        pos_encoding = positional_encoding(query)
        query = torch.cat([query, pos_encoding], dim=-1)
        key = torch.cat([self.key_conv(x), pos_encoding], dim=-1)
        value = self.value_conv(x)
        if x.is_nested:
            return self.nested_attention(query, key, value, dilation, kernel_size)
        return self.atten_func(query, key, value, kernel_size=kernel_size, dilation=dilation, is_causal=self.is_causal)

    def validate_params(self, query_params: ConvParams, key_params: ConvParams, value_params: ConvParams):
        if not check_conv_output_sizes([query_params, key_params, value_params]):
            raise ValueError("All convolutional layers must have the same output size")

    def get_dilation(self):
        if self.training:
            return self.dilation.clamp(self.min_dilation, self.max_dilation)
        else:
            return self.dilation.item()


    def get_kernel_size(self):
        if self.training:
            return self.kernel_size.clamp(self.min_kernel_size, self.max_kernel_size)
        else:
            return self.kernel_size.item()

    def nested_attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dilation: torch.Tensor, kernel_size: torch.Tensor):
        attn = self.qk_func(query, key, kernel_size= kernel_size, dilation=dilation, is_causal=self.is_causal)
        return self.v_func(attn, value, kernel_size=kernel_size, dilation=dilation, is_causal=self.is_causal)


class ConvNAT2D(ConvNAT[Conv2d]):
    conv_type = Conv2d
    atten_func = na2d
    qk_func = na2d_qk
    v_func = na2d_av

class ConvNAT3D(ConvNAT[Conv3d]):
    conv_type = Conv3d
    atten_func = na3d
    qk_func = na3d_qk
    v_func = na3d_av

