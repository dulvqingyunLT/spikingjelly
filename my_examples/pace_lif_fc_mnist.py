import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from spikingjelly.clock_driven import neuron, encoding
# from spikingjelly.clock_driven.functional import reset_net
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ops_pace import onn_fc, onn_offset, onn_octal

def reset_net(net: nn.Module):
    '''
    * :ref:`API in English <reset_net-en>`

    .. _reset_net-cn:

    :param net: 任何属于 ``nn.Module`` 子类的网络

    :return: None

    将网络的状态重置。做法是遍历网络中的所有 ``Module``，若含有 ``reset()`` 函数，则调用。

    * :ref:`中文API <reset_net-cn>`

    .. _reset_net-en:

    :param net: Any network inherits from ``nn.Module``

    :return: None

    Reset the whole network.  Walk through every ``Module`` and call their ``reset()`` function if exists.
    '''
    for m in net.modules():
        if hasattr(m, 'reset'):
            m.reset()


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 卷积层
        # self.conv1 = onn_conv2d(1, 32)
        # self.conv2 = onn_conv2d(32, 64)

        # 全连接层
        self.fc1 = onn_fc(28*28, 512)
        # self.fc1 =  nn.Linear(28 * 28, 10)
        self.lif1 = neuron.LIFNode(tau=2.0)
        self.fc2 = onn_fc(512, 10)
        # self.fc2 = nn.Linear(1024, 10)
        self.lif2 = neuron.LIFNode(tau=2.0)


        # self.onn_binary = onn_binary.apply
        self.onn_offest = onn_offset.apply
        self.onn_octal = onn_octal.apply


    def forward(self, x):        

        x = torch.flatten(x, 1)

        out_bin = self.onn_octal(x)
        out = self.fc1(out_bin)
        out = self.onn_offest(out)
        out = self.lif1(out)

        # out = self.onn_octal(out)
        out = self.fc2(out)
        out = self.onn_offest(out)
        out = self.lif2(out)


        return out

parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training')

parser.add_argument('--device', default='cpu', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

parser.add_argument('--dataset-dir', default='./', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--log-dir', default='./', help='保存tensorboard日志文件的位置，例如“./”\n Root directory for saving tensorboard logs, e.g., "./"')
parser.add_argument('--model-output-dir', default='./', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=10, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，例如“100.0”\n Membrane time constant, tau, for LIF neurons, e.g., "100.0"')
parser.add_argument('-N', '--epoch', default=10, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')


def main():
    '''
    :return: None

    * :ref:`API in English <lif_fc_mnist.main-en>`

    .. _lif_fc_mnist.main-cn:

    使用全连接-LIF的网络结构，进行MNIST识别。\n
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <lif_fc_mnist.main-cn>`

    .. _lif_fc_mnist.main-en:

    The network with FC-LIF structure for classifying MNIST.\n
    This function initials the network, starts trainingand shows accuracy on test dataset.
    '''
    
    args = parser.parse_args()
    print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("####################################")

    device = args.device
    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size 
    lr = args.lr
    T = args.T
    tau = args.tau
    train_epoch = args.epoch

    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # 定义并初始化网络
    # net = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(28 * 28, 10, bias=False),
    #     neuron.LIFNode(tau=tau)
    # )
    net = Model()
    net = net.to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()
    train_times = 0
    max_test_accuracy = 0

    test_accs = []
    train_accs = []

    """
    for epoch in range(train_epoch):
        print("Epoch {}:".format(epoch))
        print("Training...")
        train_correct_sum = 0
        train_sum = 0
        net.train()
        for img, label in tqdm(train_data_loader):
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            optimizer.zero_grad()

            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net(encoder(img).float())
                else:
                    out_spikes_counter += net(encoder(img).float())

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / T

            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            reset_net(net)

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            train_correct_sum += (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().sum().item()
            train_sum += label.numel()

            train_batch_accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
            train_accs.append(train_batch_accuracy)

            train_times += 1
        train_accuracy = train_correct_sum / train_sum

        print("Testing...")
        net.eval()
        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_correct_sum = 0
            test_sum = 0
            for img, label in tqdm(test_data_loader):
                img = img.to(device)
                for t in range(T):
                    if t == 0:
                        out_spikes_counter = net(encoder(img).float())
                    else:
                        out_spikes_counter += net(encoder(img).float())

                test_correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()
                reset_net(net)
            test_accuracy = test_correct_sum / test_sum
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            test_accs.append(test_accuracy)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        print("Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(epoch, train_accuracy, test_accuracy, max_test_accuracy, train_times))
        print()
    
    # 保存模型
    torch.save(net, model_output_dir + "/lif_snn_mnist.ckpt")
      
    """    
    # 读取模型
    net = torch.load(model_output_dir + "/lif_snn_mnist_2layer.ckpt")



    # 保存绘图用数据
    net.eval()
    with torch.no_grad():
        img, label = test_dataset[0]        
        img = img.to(device)
        for t in range(T):
            if t == 0:
                out_spikes_counter = net(encoder(img).float())    
            else:
                out_spikes_counter += net(encoder(img).float())
        out_spikes_counter_frequency = (out_spikes_counter / T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')
        output_layer = net[-1] # 输出层
        v_t_array = output_layer.v.cpu().numpy().squeeze().T  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy",v_t_array)
        s_t_array = output_layer.spike.cpu().numpy().squeeze().T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy",s_t_array)

    train_accs = np.array(train_accs)
    np.save('train_accs.npy', train_accs)
    test_accs = np.array(test_accs)
    np.save('test_accs.npy', test_accs)


def translate_model_weights(model_path):

    def round_fcw(filters):
        from ops_pace import opu
        repeats = filters.shape[0] // opu.input_vector_len
        remainder = filters.shape[0] % opu.input_vector_len
        repeats = repeats+1 if remainder!=0 else repeats
        w_scale_factor =(2 ** opu.bits - 1) / (filters.max() - filters.min() + 1e-9) 
        pad_dim=opu.input_vector_len*repeats - filters.shape[0]

        weight__ = np.round(filters * w_scale_factor )
        # weight__ = np.clip(weight__, -7, 7)
        weight__ = np.clip(weight__, -((2**(opu.bits-1))-1), ((2**(opu.bits-1))-1)) 

        weight__ = weight__.astype(np.int8)
        padded_wt = np.pad(weight__,((0,pad_dim),(0,0)),"constant",constant_values=0)
        weight_ = padded_wt.transpose(1,0).reshape(filters.shape[1], repeats, -1).transpose(2,1,0)

        return weight_

    net = Model()
    net = torch.load(model_path)
    fc1_w=net.fc1.weight.data.numpy()
    fc2_w=net.fc2.weight.data.numpy()

    fc1w = round_fcw(fc1_w)
    fc2w = round_fcw(fc2_w)

    np.savez('lif_snn_bit4_t.npz', fc1w=fc1w.transpose(2,1,0), fc2w=fc2w.transpose(2,1,0))

if __name__ == '__main__':
    main()
    # translate_model_weights('lif_snn_mnist_2layer.ckpt')
