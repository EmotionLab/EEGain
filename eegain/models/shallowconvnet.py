import torch
import torch.nn as nn

from ._registry import register_model


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


@register_model
class ShallowConvNet(nn.Module):
    @staticmethod
    def first_block(out_f, kernel_size, n_chan, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(
                1, out_f, kernel_size, padding=0, max_norm=2, *args, **kwargs
            ),
            Conv2dWithConstraint(
                40, 40, (n_chan, 1), padding=0, bias=False, max_norm=2
            ),
            nn.BatchNorm2d(out_f),
        )

    def calculate_out_size(self, n_chan, n_time):
        """
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        """

        data = torch.rand(1, 1, n_chan, n_time)
        block_one = self.firstLayer
        avg = self.avgpool
        dp = self.dp
        out = torch.log(block_one(data).pow(2))
        out = avg(out)
        out = dp(out)
        out = out.view(out.size()[0], -1)
        return out.size()

    def __init__(self, n_chan, n_time, n_class, dropout_rate, **kwargs):
        super(ShallowConvNet, self).__init__()

        kernel_size = (1, 25)
        n_filt_first_layer = 40

        self.firstLayer = ShallowConvNet.first_block(
            n_filt_first_layer, kernel_size, n_chan
        )
        self.avgpool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dp = nn.Dropout(p=dropout_rate)
        self.fSize = self.calculate_out_size(n_chan, n_time)
        self.lastLayer = nn.Linear(self.fSize[-1], n_class)

    def forward(self, x):
        x = self.firstLayer(x)
        x = torch.log(self.avgpool(x.pow(2)))
        x = self.dp(x)
        x = x.view(x.size()[0], -1)
        x = self.lastLayer(x)

        return x
