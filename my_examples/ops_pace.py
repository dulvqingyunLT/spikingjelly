import numpy as np
# from sqlalchemy import true

# import tensorflow as tf
# from onn_pace import OPU
# from gen_spec import *

import torch
import torch.nn  as nn
# from omegaconf import OmegaConf
import numpy as np
import torch.nn.functional as F
import math

# pace_spec = OmegaConf.load('pace_spec.yaml')
# opu = OPU(pace_spec)

class OPU:
    """ Optical Processing Unit (OPU) Design """

    def __init__(self):
        self.input_vector_len = 64
        self.weight_vector_len = 64
        self.bits = 4
        self.out_bits = 4
        self.tia_noise_mean = 0.0
        self.tia_noise_sigma = 1.0

# pace_spec = OmegaConf.load('pace_spec.yaml')
opu = OPU()

class onn_round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.round(input)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class onn_octal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.round(input*255) #input范围[0,1]      
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class onn_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        thresh =  torch.min(input) + (torch.max(input) - torch.min(input)) /2
        output = torch.where(input<=thresh, 0.0, 1.0)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class onn_thresh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = torch.where(input<0, 0.0, 1.0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class onn_offset(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        average =  torch.min(input) + (torch.max(input) - torch.min(input)) /2
        output = input - average
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class onn_dot_fun(torch.autograd.Function):
    def __init__(self, opu) -> None:
        super().__init__()

    cv_bits = opu.bits
    tia_noise_sigma = opu.tia_noise_sigma
    tia_noise_mean = opu.tia_noise_mean
    out_bits = opu.out_bits
    # my_onn_round = onn_round.apply
    onn_thresh = onn_thresh.apply

    @staticmethod
    def forward(ctx, input, weight, ):
        ctx.save_for_backward(input, weight)
        out_mat = input.matmul(weight)
        out_mat = torch.clamp(out_mat, -128, 127)
        # thresh = torch.randn(size=out_mat.size(), requires_grad=False, device=input.device)*onn_dot_fun.tia_noise_sigma + onn_dot_fun.tia_noise_mean
        # out_mat += thresh
        # out_interm = onn_dot_fun.my_onn_round(out_mat) # input和weight均为整数，所以不需要round操作了。
        result = onn_dot_fun.onn_thresh(out_mat)
        # bit_shift_result = out_interm.type(torch.uint8) >> (8 - onn_dot_fun.out_bits)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight.t())
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input).t()

        return grad_input, grad_weight

class onn_conv2d(nn.Module):
    def __init__(self, input_channel, output_channel, filter_height=3, filter_width=3, stride=1, padding='SAME', add_bias=False,
           hardware=opu):
        super(onn_conv2d, self).__init__()

        self.output_channel = output_channel
        self.input_channel = input_channel
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.padding = padding
        self.hardware = hardware

        # wt_array = np.random.rand(low=-2**(self.hardware.bits-1)-1, high=2**(self.hardware.bits-1), 
        #                             size=(output_channel,input_channel,filter_height,filter_width))
        self.weight = nn.Parameter(torch.Tensor(output_channel,input_channel,filter_height,filter_width))
        if add_bias:
            # bias_array = np.random.randint(low=-2**(self.hardware.bits-1)-1, high=2**(self.hardware.bits-1),size=(output_channel))
            self.bias = nn.Parameter(torch.Tensor(output_channel))
        else:
            self.bias = None
        

        self.onn_round = onn_round.apply
        self.onn_dot_fun = onn_dot_fun.apply

        assert(filter_height==filter_width)
        assert(padding=='SAME')

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, inp): # [b c h w]
        
        k = self.filter_width
        stride = self.stride
        batch_size,c_in,h_in, w_in = inp.size()
        assert(c_in == self.input_channel)

        # padding = self.padding  # + k//2
        padding = (k-stride)//2 if self.padding=='SAME' else 0


        h_out = (h_in + 2 * padding - (k - 1) - 1) / stride + 1
        w_out = (w_in + 2 * padding - (k - 1) - 1) / stride + 1
        h_out, w_out = int(h_out), int(w_out)

        w_scale_factor =(2**self.hardware.bits-1) / (self.weight.max()-self.weight.min() + 1e-9) 

        inp_unf_ = torch.nn.functional.unfold(inp, (k, k), padding=padding)  #[b, c*k*k, L]
        #c*k*k表示有多少个局部块，L表示局部块的大小
        inp_unf_=inp_unf_.transpose(1, 2)  #[b, L, c*k*k]
        # with torch.no_grad():
        # weight__ = self.onn_round( (self.weight + self.w_zero_point)* self.w_scale_factor) # 先对权值进行取整
        weight__ = self.onn_round(self.weight * w_scale_factor )
        weight__ = torch.clamp(weight__, -((2**(self.hardware.bits-1))-1), ((2**(self.hardware.bits-1))-1))    

        weight_ = weight__.view(self.output_channel, -1).t() #[out_c,c*k*k]-->[c*k*k, out_c]

        repeats = inp_unf_.size(-1) // self.hardware.input_vector_len
        remainder = inp_unf_.size(-1) % self.hardware.input_vector_len
        repeats = repeats+1 if remainder!=0 else repeats

        dim=(0, self.hardware.input_vector_len*repeats - weight_.size(0), 0, 0) #左右上下， 填右边
        zero_tensor_0=F.pad(inp_unf_,dim,"constant",value=0)
 
        inp_unf = zero_tensor_0.contiguous().view([batch_size*inp_unf_.size(1), repeats, -1])

        dim=(0, 0, 0,self.hardware.input_vector_len*repeats - weight_.size(0)) #左右上下， 填下边
        zero_tensor_1=F.pad(weight_,dim,"constant",value=0)

        weight = zero_tensor_1.permute(1,0).reshape(weight_.size(1), repeats, -1).permute(2,1,0)

        temp_out = torch.zeros([batch_size*inp_unf_.size(1),weight_.size(1)], device=inp.device)
        for i in range(repeats):
            matmul_result = self.onn_dot_fun(inp_unf[:, i, :], weight[:,i,:])
            temp_out = temp_out+ matmul_result

        out_ = temp_out.view([batch_size,inp_unf_.size(1),weight_.size(1)])
        
        out_unf = out_.transpose(1,2)

        out = torch.nn.functional.fold(out_unf, (h_out, w_out), (1, 1))

        return out


class onn_fc(nn.Module):
    def __init__(self, input_dimension, output_dimension, hardware=opu, add_bias=False) -> None:
        super(onn_fc, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hardware = hardware
        # self.onn_activation = onn_activation(threshold=1)
        self.add_bias = add_bias

        # wt_array = np.random.randint(low=-2**(self.hardware.bits-1)-1, high=2**(self.hardware.bits-1), 
        #                             size=(input_dimension, output_dimension))
        self.weight = nn.Parameter(torch.Tensor(input_dimension, output_dimension))
        if add_bias:
            # bias_array = np.random.randint(low=-2**(self.hardware.bits-1)-1, high=2**(self.hardware.bits-1),size=(output_dimension))
            self.bias = nn.Parameter(torch.Tensor(output_dimension))
        else:
            self.bias = None
     
        self.onn_round = onn_round.apply
        self.onn_dot_fun = onn_dot_fun.apply

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inp):

        assert(inp.size(-1) == self.input_dimension)
        batch_size = inp.size(0)

        repeats = inp.size(-1) // self.hardware.input_vector_len
        remainder = inp.size(-1) % self.hardware.input_vector_len
        repeats = repeats+1 if remainder!=0 else repeats
        
        w_scale_factor =(2**self.hardware.bits-1) / (self.weight.max()-self.weight.min() + 1e-9) 

        
        dim=(0,self.hardware.input_vector_len*repeats-inp.size(-1),0, 0) #左右上下, 填右边
        inp_=F.pad(inp,dim,"constant",value=0)

        inp_ = inp_.contiguous().view(batch_size, repeats, -1)

        weight__ = self.onn_round(self.weight * w_scale_factor )

        weight__ = torch.clamp(weight__, -((2**(self.hardware.bits-1))-1), ((2**(self.hardware.bits-1))-1)) 

        dim=(0, 0, 0,self.hardware.input_vector_len*repeats-inp.size(-1)) #左右上下， 填下边
        weight_=F.pad(weight__,dim,"constant",value=0)

        weight_ = weight_.permute(1,0).reshape(weight__.size(1), repeats, -1).permute(2,1,0)

        temp_out = torch.zeros([batch_size, weight__.size(1)], device=inp.device)
        for i in range(repeats):
            matmul_result = self.onn_dot_fun(inp_[:, i, :], weight_[:,i,:])
            temp_out = temp_out+ matmul_result
        
        if self.add_bias:
            temp_out = temp_out + self.bias




