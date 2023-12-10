import torch.nn as nn
import math

class ChapterAdapter(nn.Module):
    # NOTE: The paper arXiv:2212.01282 uses layer norm instead of batch norm,
    # but since dimensions weren' specified and code isn't available I've opted for the latter
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 compression_rate = 4,):
        assert out_channels % compression_rate == 0
        super(ChapterAdapter, self).__init__()

        self.compression_rate = compression_rate
        self.conv_delta_fn = nn.Conv2d(in_channels  = in_channels,
                                        out_channels = out_channels // compression_rate,
                                        kernel_size  = kernel_size,
                                        stride       = stride,
                                        bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels // compression_rate)
        self.activation = nn.GELU(approximate='tanh')

        self._init_weight()
    
    def _init_weight(self):
        conv = self.conv_delta_fn
        n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels

        self.conv_delta_fn.weight.data.normal_(0, math.sqrt(2. / n))

        self.batch_norm.weight.data.fill_(1)
        self.batch_norm.bias.data.zero_()
        
    def forward(self, x):
        out = self.conv_delta_fn(x)
        out = self.batch_norm(out)
        out = self.activation(out)
        out = out.repeat(1, self.compression_rate, 1, 1)
        return out

class RebuffiAdapter(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride = 1):
        super(RebuffiAdapter, self).__init__()
        self.conv_delta_fn = nn.Conv2d(in_channels  = in_channels,
                                       out_channels = out_channels,
                                       kernel_size  = 1,
                                       stride       = stride,
                                       bias = False)
        self.batch_norm_1 = nn.BatchNorm2d(in_channels)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)

        self._init_weight()

    def _init_weight(self):
        conv = self.conv_delta_fn
        n = conv.kernel_size[0] * conv.kernel_size[1] * conv.out_channels

        self.conv_delta_fn.weight.data.normal_(0, math.sqrt(2. / n))

        self.batch_norm_1.weight.data.fill_(1)
        self.batch_norm_1.bias.data.zero_()
        
        self.batch_norm_2.weight.data.fill_(1)
        self.batch_norm_2.bias.data.zero_()

    def forward(self,x):
        delta_x = self.conv_delta_fn(self.batch_norm_1(x))
        out = self.batch_norm_2(x + delta_x)
        return out