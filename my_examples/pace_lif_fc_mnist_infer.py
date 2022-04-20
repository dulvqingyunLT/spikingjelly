from cgitb import lookup
from genericpath import isfile


import numpy as np
import logging 
import time
import os
import random

from torch import channels_last



class OPU:
    """ Optical Processing Unit (OPU) Design """

    def __init__(self):
        self.input_vector_len = 64
        self.weight_vector_len = 64
        self.bits = 4
        self.out_bits = 4
        self.tia_noise_mean = 0.0
        self.tia_noise_sigma = 1.0


opu = OPU()
onn_out_table = dict()


var_name = ['fc1w',
            'fc2w',]

def extract_npy(var_path):

    var_dict = np.load(var_path)
    arr_list = []
    for i in var_name:
        arr_list += [var_dict[i]]

    return arr_list



"""
weight: 64 elements vector
input: 64 elements vector
weight_fix_flag: wether weight changed compared to latest call, defalut is True
"""
def onn_dot_sim(weight, input, weight_fix_flag=True):
    
    out_mat = np.dot(weight,input)
    out_mat = np.clip(out_mat, -128, 127) #+ np.random.normal(size=out_mat.shape)

    # result = np.where(out_mat<0, 0, 1).astype(np.uint8)
    result = 0 if out_mat<0 else 1

    return np.uint8(result)

# def onn_binary(input):
#     thresh =  np.min(input) + (np.max(input) - np.min(input)) / 2
#     output = np.where(input<=thresh, 0.0, 1.0)
#     return output.astype(np.uint8)

def onn_offset(input):

    average =  np.min(input) + (np.max(input) - np.min(input)) /2
    output = input - average
        
    return output

def onn_octary(input):
    thresh =  np.min(input) + (np.max(input) - np.min(input)) / 2
    output = np.where(input<=thresh, 0.0, 1.0)
    return output.astype(np.uint8)

class LIFNode():
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
        v_reset: float = 0., detach_reset: bool = False) -> None:

        assert isinstance(tau, float) and tau > 1.
        self.tau = tau
        self.decay_input = decay_input
        self.v_reset = v_reset
        self.v_threshold = v_threshold
        self.v = 0
        self.detach_reset = detach_reset
    
    def neuronal_fire(self): # using heaviside function in forward

        input = self.v - self.v_threshold
        return np.where(input>=0, 1, 0)
    


    def neuronal_charge(self, x):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) / self.tau
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) / self.tau

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - 1. / self.tau) + x
            else:
                self.v = self.v - (self.v - self.v_reset) / self.tau + x

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - spike_d * self.v_threshold

        else:
            # hard reset
            self.v = (1. - spike_d) * self.v + spike_d * self.v_reset


    def forward(self,x):
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def reset_node(self):

        self.v = 0
        
def img_unfold(input_data, filter_h, filter_w, stride=1, pad=0, channelast=False): # input_data batch为1
    #inp_unf_ = torch.nn.functional.unfold(inp, (k, k), padding=padding)  #[b, c*k*k, L]
    assert(stride==1)
    if channelast:
        H, W, C = input_data.shape
        input_data = np.transpose(input_data, [2, 0, 1])
    else:
        C, H, W = input_data.shape

    img = np.pad(input_data, [(0, 0), (pad, pad), (pad, pad)], 'constant')
    _, new_img_h, new_img_w = img.shape

    out_dim_h = (new_img_h - filter_h + 1)//stride
    out_dim_w = (new_img_w - filter_w +1)//stride

    out = np.zeros((out_dim_h*out_dim_w, C, filter_h, filter_w),dtype=input_data.dtype)

    for y in range(0, new_img_h - filter_h + 1, stride):

        for x in range(0, new_img_w - filter_w +1, stride):

            out[y*out_dim_h+x] = img[:,y:y+filter_h, x:x+filter_w]

    return out.reshape(out_dim_h*out_dim_w,-1)

def fc_infer(x, filters):
    batch_size = x.shape[0]
    repeats = x.shape[-1] // opu.input_vector_len
    remainder = x.shape[-1] % opu.input_vector_len
    repeats = repeats+1 if remainder!=0 else repeats
    
    pad_dim=opu.input_vector_len*repeats-x.shape[-1]#填右边
    inp_ = np.pad( x, ((0,0),(0, pad_dim)),"constant",constant_values=0)

    inp_ = inp_.reshape([batch_size, repeats, opu.input_vector_len]).transpose(2,1,0)
    key_mat = np.matmul(filters.transpose(1,0,2),inp_.transpose(1,0,2)).transpose(1,0,2)
    temp_out = np.zeros(shape=(filters.shape[0], repeats, x.shape[0]), dtype=np.int32)
    key_max = key_mat.max()
    key_min = key_mat.min()

    for key in range(key_min,key_max+1):
        index = np.where(key_mat==key)  #取三个array的第一个元素，分别对应n,i,m
        if not index[0].any():
            continue
        if key  not in onn_out_table:

            temp_value = onn_dot_sim(filters[index[0][0],index[1][0],:], inp_[:,index[1][0],index[2][0]])
            onn_out_table[key] = temp_value  
        temp_out[index[0],index[1],index[2]]+=onn_out_table[key]
    temp_out = np.sum(temp_out,axis=1,keepdims=False)
    return temp_out.transpose(1,0)

def conv2d_infer(x, filters, channelast=False):
    stride = 1
    padding = 1
    filter_h = 3
    filter_w = 3
    c_in, h_in, w_in = x.shape

    h_out = (h_in + 2 * padding - (filter_h - 1) - 1) / stride + 1
    w_out = (w_in + 2 * padding - (filter_h - 1) - 1) / stride + 1
    h_out, w_out = int(h_out), int(w_out)

    # st_time = time.time()
    inp_unf_ = img_unfold(x, filter_h, filter_w, stride=1, pad=padding, channelast=channelast)
    # end_time = time.time()
    # print("time:{}".format((end_time-st_time)*1000))
    repeats = inp_unf_.shape[-1] // opu.input_vector_len
    remainder = inp_unf_.shape[-1] % opu.input_vector_len
    repeats = repeats+1 if remainder!=0 else repeats

    pad_dim = opu.input_vector_len*repeats - filter_h*filter_w*c_in
    padded_inp = np.pad( inp_unf_, ((0,0),(0,pad_dim)),"constant",constant_values=0)

    inp_unf = padded_inp.reshape([inp_unf_.shape[0], repeats, opu.input_vector_len]).transpose(2,1,0)
    temp_out = np.zeros(shape=(filters.shape[0], repeats, inp_unf.shape[-1]), dtype=np.int32)

    key_mat = np.matmul(filters.transpose(1,0,2),inp_unf.transpose(1,0,2)).transpose(1,0,2)
    key_max = key_mat.max()
    key_min = key_mat.min()
    for key in range(key_min,key_max+1):
        index = np.where(key_mat==key)  #取三个array的第一个元素，分别对应n,i,m
        if not index[0].any():
            continue
        if key not in onn_out_table:
            first_input = inp_unf[:,index[1][0],index[2][0]]
            if(not first_input.any()):# 输入全零的话，不计算直接赋值
                onn_out_table[key] = 1
            else:
                temp_value = onn_dot_sim(filters[index[0][0],index[1][0],:], inp_unf[:,index[1][0],index[2][0]])        
                onn_out_table[key] = temp_value
        temp_out[index[0],index[1],index[2]]+=onn_out_table[key]

    temp_out = np.sum(temp_out,axis=1,keepdims=False)
    out = temp_out.reshape([filters.shape[0], h_out, w_out])
    
    return out

def maxpooling(x, pool_size=2):
    pool_out = x.reshape(x.shape[0], x.shape[1]//pool_size, pool_size, x.shape[2]//pool_size, pool_size)
    pool_out = pool_out.max(axis=(2, 4))
    return pool_out
    
def translate_mnist():
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    batch=64
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=torchvision.transforms.Compose([
                                          torchvision.transforms.ToTensor(),  # 转换成张量
                                          torchvision.transforms.Lambda(lambda x: torch.round(x*255)  ),
                                        #   torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 标准化
                                      ]))
    test_loader = DataLoader(test, batch_size=batch)  # 分割训练
    test_images = torch.empty((1,1,28,28),dtype=torch.float32)
    test_labels = torch.empty((1),dtype=torch.int64)
    for x, y in test_loader:
        test_images = torch.cat((test_images,x),dim=0)
        test_labels = torch.cat((test_labels,y),dim=0)
    test_images = test_images[1:,...]
    test_labels = test_labels[1:,...]
    np.save("./data/test_images", test_images.numpy())
    np.save("./data/test_labels", test_labels.numpy())



if __name__ == "__main__":

    var_path = './lif_snn_bit4_t.npz' # 第二个卷积输出64维
    vars = extract_npy(var_path)

    fc1_w = vars[0]
    fc2_w = vars[1]

    if not os.path.isfile("/home/wkuang/snn/spikingjelly/data/test_images.npy") or not isfile("/home/wkuang/snn/spikingjelly/data/test_labels.npy"):
        translate_mnist()

    test_data = np.load("./data/test_images.npy")
    test_labels = np.load("./data/test_labels.npy")

    import time
    

    infer_labels = []
    lif1 = LIFNode()
    lif2 = LIFNode()

    for image, label in zip(test_data, test_labels):
        st_time = time.time()
        onn_out_table.clear()
        infer_label = np.zeros(shape=(1,10))

        for t in range(100):
            input = np.random.randint(low=0, high=256,  size=image.shape)
            input = np.where(input<image, input, 0)
            out = np.expand_dims(input.flatten(),0)

            out = fc_infer(out, fc1_w)
            out = onn_offset(out)
            out = lif1.forward(out)

            out = fc_infer(out, fc2_w)
            out = onn_offset(out)
            out = lif2.forward(out)

            infer_label += out
        
        lif1.reset_node()
        lif2.reset_node()
        infer_label = np.argmax(infer_label)
        end_time = time.time()
        
        print("infer: {}, label: {}, time: {} ms".format(infer_label, label, (end_time-st_time)*1000))
        infer_labels += [infer_label]

    infer_labels = np.asarray(infer_labels)
    accuracy = np.mean(np.equal(infer_labels, test_labels))
    print("accuracy:", accuracy)
    print("fps: ", test_labels.size / (time.time() - st_time))

