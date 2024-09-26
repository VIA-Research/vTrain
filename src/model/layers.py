import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from .utils import divide
from .utils import split_tensor_along_last_dim


class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, world_size,
                 bias=True, gather_output=True, skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.world_size = world_size
        # Divide the weight matrix along the last dimension.
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # self.linear = nn.Linear(input_size, self.output_size_per_partition, bias=bias)
        self.weight = Parameter(torch.empty(self.output_size_per_partition, input_size))
        self.bias = Parameter(torch.empty(self.output_size_per_partition))

    def forward(self, input_):
        # Matrix multiply.
        # output_parallel = self.linear(input_)
        bias = self.bias if not self.skip_bias_add else None
        # print (input_.shape, self.weight.shape, bias)
        # exit()
        output_parallel = F.linear(input_, self.weight, bias)
        if self.gather_output:
            output = torch.cat([output_parallel for _ in range(self.world_size)], -1) 
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, world_size,
                 bias=True, input_is_parallel=False, stride=1, skip_bias_add=True):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        self.world_size = world_size
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # self.linear = nn.Linear(self.input_size_per_partition, output_size, bias=bias)
        self.weight = Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.bias = Parameter(torch.empty(self.output_size))

    def forward(self, input_):
        if not self.input_is_parallel:
            input_parallel, *_ = split_tensor_along_last_dim(input_, self.world_size)
        else:
            input_parallel = input_
            
        # Matrix multiply.
        # output = self.linear(input_parallel)
        output_ = F.linear(input_parallel, self.weight)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

