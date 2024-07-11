from collections import namedtuple
import torch
from torch import nn

WIDTH = 4
HEIGHT = 4
INPUT_SIZE = (25, WIDTH, HEIGHT)
EXTRA_SIZE = 14
V_SIZE = 2
PI_SIZE = 1024

NN_DEPTH = 8
NN_CHANNELS = 32
NN_LR = 0.001
NN_LR_MILESTONE = 500

NNArgs = namedtuple(
    "NNArgs",
    [
        "v_size",
        "pi_size",
        "num_channels",
        "depth",
        "lr_milestone",
        "lr",
        "cv",
    ],
)

def conv(in_channels, out_channels, stride=1, kernel_size=3):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding="same",
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    return conv(in_channels, out_channels, stride, 1)


def conv3x3(in_channels, out_channels, stride=1):
    return conv(in_channels, out_channels, stride, 3)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size=4):
        super(DenseBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.Mish(inplace=True)
        self.conv1 = conv1x1(in_channels, growth_rate * bn_size)
        self.bn2 = nn.BatchNorm2d(growth_rate * bn_size)
        self.relu2 = nn.Mish(inplace=True)
        self.conv2 = conv3x3(growth_rate * bn_size, growth_rate)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)
        out = torch.cat([x, out], 1)
        return out


class NNArch(nn.Module):
    def __init__(self, args):
        super(NNArch, self).__init__()

        in_channels, in_x, in_y = INPUT_SIZE
        if EXTRA_SIZE:
            in_channels -= 1

        layers1 = []
        for i in range(args.depth // 2):
            layers1.append(
                DenseBlock(
                    in_channels + args.num_channels * i,
                    args.num_channels,
                )
            )
        self.conv_layers1 = nn.Sequential(*layers1)

        global_pooling_c = in_channels + args.num_channels * (args.depth // 2)
        self.global_pooling_bn = nn.BatchNorm2d(global_pooling_c)
        self.global_pooling_relu = nn.Mish(inplace=True)
        self.global_pooling_fc = nn.Linear(global_pooling_c * 2 + EXTRA_SIZE, global_pooling_c)

        layers2 = []
        for i in range(args.depth // 2, args.depth):
            layers2.append(
                DenseBlock(
                    in_channels + args.num_channels * i,
                    args.num_channels,
                )
            )
        self.conv_layers2 = nn.Sequential(*layers2)


        final_size = in_channels + args.num_channels * args.depth
        self.v_conv = conv1x1(final_size, 32)
        self.pi_conv = conv1x1(final_size, 32)

        self.v_bn = nn.BatchNorm2d(32)
        self.v_relu = nn.Mish(inplace=True)
        self.v_flatten = nn.Flatten()
        self.v_fc1 = nn.Linear(32 * in_x * in_y + EXTRA_SIZE, 256)
        self.v_fc1_relu = nn.Mish(inplace=True)
        self.v_fc2 = nn.Linear(256, args.v_size)
        self.v_softmax = nn.LogSoftmax(1)

        self.pi_bn = nn.BatchNorm2d(32)
        self.pi_relu = nn.Mish(inplace=True)
        self.pi_flatten = nn.Flatten()
        self.pi_fc1 = nn.Linear(32 * in_x * in_y + EXTRA_SIZE, args.pi_size)
        self.pi_softmax = nn.LogSoftmax(1)

    # s = batch_size * num_channels * board_x * board_y
    def forward(self, s):
        s, l = torch.split(s, [INPUT_SIZE[0]-1, 1], dim=1)
        l = torch.flatten(l, start_dim=1)[:, :EXTRA_SIZE]

        s = self.conv_layers1(s)

        # global pooling
        x = self.global_pooling_bn(s)
        x = self.global_pooling_relu(x)
        x = x.view(-1, INPUT_SIZE[0]-1+NN_CHANNELS*(NN_DEPTH//2), INPUT_SIZE[1] * INPUT_SIZE[2])
        y = torch.mean(x, dim=2)
        z, _ = torch.max(x, dim=2)
        s = s + self.global_pooling_fc(torch.cat([y, z, l], 1)).unsqueeze(-1).unsqueeze(-1)

        s = self.conv_layers2(s)

        v = self.v_conv(s)
        v = self.v_bn(v)
        v = self.v_relu(v)
        v = self.v_flatten(v)
        v = torch.cat((v, l), 1)
        v = self.v_fc1(v)
        v = self.v_fc1_relu(v)
        v = self.v_fc2(v)
        v = self.v_softmax(v)

        pi = self.pi_conv(s)
        pi = self.pi_bn(pi)
        pi = self.pi_relu(pi)
        pi = self.pi_flatten(pi)
        pi = torch.cat((pi, l), 1)
        pi = self.pi_fc1(pi)
        pi = self.pi_softmax(pi)

        return v, pi
